//! Module A native operators.
//!
//! This crate contains the source-owned scalar APIs plus the profiler-backed
//! A3b per-frame candidate batch API. Detector wiring remains outside this
//! crate's contract.

use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

const API_VERSION: &str = "1";
const A3B_BATCH_API_VERSION: &str = "1";
const CRATE_VERSION: &str = env!("CARGO_PKG_VERSION");
const SOURCE_MANIFEST_SHA256: &str = env!("MODULE_A_NATIVE_SOURCE_MANIFEST_SHA256");
const MAX_SAFE_COORD: i64 = i32::MAX as i64;
const CAPABILITIES: [&str; 6] = [
    "a1_lbp_features",
    "a2_change_features",
    "best_grid_value_f32",
    "a3b_boxes_stats",
    "a3b_one_box_stats",
    "blinding_laplacian_var",
];

type A3bBoxStats = (f64, f64, f64, f64, f64, f64);

#[inline(always)]
fn hist_lbp(
    view: &numpy::ndarray::ArrayView2<'_, u8>,
    x1: usize,
    y1: usize,
    x2: usize,
    y2: usize,
) -> [f32; 32] {
    let mut counts = [0u64; 32];
    for yy in y1..y2 {
        for xx in x1..x2 {
            let value = unsafe { *view.uget((yy, xx)) };
            counts[(value >> 3) as usize] += 1;
        }
    }
    let mut out = [0f32; 32];
    let total: u64 = counts.iter().sum();
    if total == 0 {
        return out;
    }
    let total_f32 = total as f32;
    for index in 0..32 {
        out[index] = counts[index] as f32 / total_f32;
    }
    out
}

#[inline(always)]
fn hist_distance(a: &[f32; 32], b: &[f32; 32]) -> f32 {
    let mut sum = 0f32;
    for index in 0..32 {
        sum += (a[index] - b[index]).abs();
    }
    0.5 * sum
}

#[inline(always)]
fn clamp_i64(value: i64, low: i64, high: i64) -> i64 {
    if value < low {
        low
    } else if value > high {
        high
    } else {
        value
    }
}

fn validate_safe_rois(rois: &[(i64, i64, i64, i64)], argument: &str) -> PyResult<()> {
    for (index, &(x1, y1, x2, y2)) in rois.iter().enumerate() {
        if [x1, y1, x2, y2]
            .iter()
            .any(|value| value.unsigned_abs() > MAX_SAFE_COORD as u64)
        {
            return Err(PyValueError::new_err(format!(
                "{argument}[{index}] contains a coordinate outside the supported i32 range"
            )));
        }
    }
    Ok(())
}

fn validate_same_shape(
    left_name: &str,
    left_shape: &[usize],
    right_name: &str,
    right_shape: &[usize],
) -> PyResult<()> {
    if left_shape != right_shape {
        return Err(PyValueError::new_err(format!(
            "{left_name} and {right_name} must have the same shape, got \
             {left_shape:?} and {right_shape:?}"
        )));
    }
    Ok(())
}

fn validate_box(
    x1: i64,
    y1: i64,
    x2: i64,
    y2: i64,
    width: usize,
    height: usize,
) -> PyResult<(usize, usize, usize, usize)> {
    let width_i64 = i64::try_from(width)
        .map_err(|_| PyValueError::new_err("array width exceeds the supported i64 range"))?;
    let height_i64 = i64::try_from(height)
        .map_err(|_| PyValueError::new_err("array height exceeds the supported i64 range"))?;
    if x1 < 0 || y1 < 0 || x2 <= x1 || y2 <= y1 || x2 > width_i64 || y2 > height_i64 {
        return Err(PyValueError::new_err(format!(
            "bbox must satisfy 0 <= x1 < x2 <= {width_i64} and \
             0 <= y1 < y2 <= {height_i64}, got ({x1}, {y1}, {x2}, {y2})"
        )));
    }
    Ok((x1 as usize, y1 as usize, x2 as usize, y2 as usize))
}

/// Compute A1 LBP histogram aggregates.
///
/// Returns `(delta_h_global, delta_h_local_max, local_mean, local_box,
/// delta_h_roi_max, delta_h_target_contrast, delta_h_roi_patch_max,
/// target_box)`.
#[pyfunction]
#[pyo3(signature = (lbp, rois, baseline=None))]
fn a1_lbp_features(
    py: Python<'_>,
    lbp: PyReadonlyArray2<u8>,
    rois: Vec<(i64, i64, i64, i64)>,
    baseline: Option<PyReadonlyArray1<f32>>,
) -> PyResult<PyObject> {
    validate_safe_rois(&rois, "rois")?;
    let view = lbp.as_array();
    let height = view.shape()[0];
    let width = view.shape()[1];

    let global_hist = hist_lbp(&view, 0, 0, width, height);
    let baseline_hist: [f32; 32] = match baseline {
        Some(array) => {
            let values = array.as_array();
            if values.len() != 32 {
                return Err(PyValueError::new_err(format!(
                    "baseline must contain exactly 32 float32 values, got {}",
                    values.len()
                )));
            }
            let mut out = [0f32; 32];
            for index in 0..32 {
                out[index] = values[index];
            }
            out
        }
        None => global_hist,
    };

    let delta_h_global = hist_distance(&global_hist, &baseline_hist) as f64;

    let grid: usize = 8;
    let cell_width = std::cmp::max(16, width / grid);
    let cell_height = std::cmp::max(16, height / grid);
    let mut local_max = -1.0f64;
    let mut local_box = (0i64, 0i64, width as i64, height as i64);
    let mut local_sum = 0.0f64;
    let mut local_count = 0u64;
    let mut y = 0usize;
    while y < height {
        let box_y2 = std::cmp::min(height, y + cell_height);
        let mut x = 0usize;
        while x < width {
            let box_x2 = std::cmp::min(width, x + cell_width);
            let local_hist = hist_lbp(&view, x, y, box_x2, box_y2);
            let distance = hist_distance(&local_hist, &global_hist) as f64;
            local_sum += distance;
            local_count += 1;
            if distance > local_max {
                local_max = distance;
                local_box = (x as i64, y as i64, box_x2 as i64, box_y2 as i64);
            }
            x += cell_width;
        }
        y += cell_height;
    }
    let delta_h_local_max = if local_max < 0.0 { 0.0 } else { local_max };
    let local_mean = if local_count > 0 {
        local_sum / local_count as f64
    } else {
        0.0
    };

    let mut delta_h_roi_max = 0.0f64;
    let mut delta_h_target_contrast = 0.0f64;
    let mut delta_h_roi_patch_max = 0.0f64;
    let mut target_box: Option<(i64, i64, i64, i64)> = None;

    for (raw_x1, raw_y1, raw_x2, raw_y2) in rois.iter().copied() {
        // Existing detector semantics clip ordinary out-of-frame coordinates.
        let x1 = clamp_i64(raw_x1, 0, width as i64);
        let y1 = clamp_i64(raw_y1, 0, height as i64);
        let x2 = clamp_i64(raw_x2, 0, width as i64);
        let y2 = clamp_i64(raw_y2, 0, height as i64);
        if x2 <= x1 || y2 <= y1 {
            continue;
        }

        let roi_hist = hist_lbp(&view, x1 as usize, y1 as usize, x2 as usize, y2 as usize);
        let contrast = hist_distance(&roi_hist, &global_hist) as f64;
        let baseline_contrast = hist_distance(&roi_hist, &baseline_hist) as f64;
        let roi_score = contrast.max(baseline_contrast);

        let sub_width = std::cmp::max(8, (raw_x2 - raw_x1) / 4);
        let sub_height = std::cmp::max(8, (raw_y2 - raw_y1) / 4);
        let mut sub_y = raw_y1;
        while sub_y < raw_y2 {
            let mut sub_x = raw_x1;
            while sub_x < raw_x2 {
                let patch_x2 = std::cmp::min(raw_x2, sub_x.saturating_add(sub_width));
                let patch_y2 = std::cmp::min(raw_y2, sub_y.saturating_add(sub_height));
                if patch_x2 - sub_x < 8 || patch_y2 - sub_y < 8 {
                    sub_x = sub_x.saturating_add(sub_width);
                    continue;
                }
                let clipped_x1 = clamp_i64(sub_x, 0, width as i64) as usize;
                let clipped_y1 = clamp_i64(sub_y, 0, height as i64) as usize;
                let clipped_x2 = clamp_i64(patch_x2, 0, width as i64) as usize;
                let clipped_y2 = clamp_i64(patch_y2, 0, height as i64) as usize;
                if clipped_x2 > clipped_x1 && clipped_y2 > clipped_y1 {
                    let patch_hist =
                        hist_lbp(&view, clipped_x1, clipped_y1, clipped_x2, clipped_y2);
                    let patch_score = hist_distance(&patch_hist, &roi_hist)
                        .max(hist_distance(&patch_hist, &baseline_hist))
                        .max(hist_distance(&patch_hist, &global_hist))
                        as f64;
                    if patch_score > delta_h_roi_patch_max {
                        delta_h_roi_patch_max = patch_score;
                    }
                }
                sub_x = sub_x.saturating_add(sub_width);
            }
            sub_y = sub_y.saturating_add(sub_height);
        }

        if roi_score > delta_h_roi_max {
            delta_h_roi_max = roi_score;
            delta_h_target_contrast = contrast;
            target_box = Some((raw_x1, raw_y1, raw_x2, raw_y2));
        }
    }

    let target_box_object: PyObject = match target_box {
        Some(value) => value.into_py(py),
        None => py.None(),
    };
    Ok((
        delta_h_global,
        delta_h_local_max,
        local_mean,
        local_box,
        delta_h_roi_max,
        delta_h_target_contrast,
        delta_h_roi_patch_max,
        target_box_object,
    )
        .into_py(py))
}

/// Compute A2 temporal LBP change aggregates.
///
/// Returns `(change_t_global, change_t_local_max, change_t_local_mean,
/// local_box, change_t_roi_max, target_box, change_t_context_mean)`.
#[pyfunction]
fn a2_change_features(
    py: Python<'_>,
    lbp: PyReadonlyArray2<u8>,
    prev_lbp: PyReadonlyArray2<u8>,
    rois: Vec<(i64, i64, i64, i64)>,
    expand_margin: f64,
) -> PyResult<PyObject> {
    validate_safe_rois(&rois, "rois")?;
    if !expand_margin.is_finite() || expand_margin < 0.0 {
        return Err(PyValueError::new_err(format!(
            "expand_margin must be finite and >= 0, got {expand_margin}"
        )));
    }

    let current = lbp.as_array();
    let previous = prev_lbp.as_array();
    validate_same_shape("lbp", current.shape(), "prev_lbp", previous.shape())?;
    let height = current.shape()[0];
    let width = current.shape()[1];
    let pixel_count = width
        .checked_mul(height)
        .ok_or_else(|| PyValueError::new_err("lbp shape is too large"))?;

    let mut diff = vec![0f32; pixel_count];
    for y in 0..height {
        let row = y * width;
        for x in 0..width {
            let current_value = unsafe { *current.uget((y, x)) } as i32;
            let previous_value = unsafe { *previous.uget((y, x)) } as i32;
            diff[row + x] = (current_value - previous_value).abs() as f32 / 255.0f32;
        }
    }

    let change_t_global = if pixel_count > 0 {
        diff.iter().map(|&value| value as f64).sum::<f64>() / pixel_count as f64
    } else {
        0.0
    };

    let grid: usize = 8;
    let cell_width = std::cmp::max(8, width / grid);
    let cell_height = std::cmp::max(8, height / grid);
    let mut best = 0f64;
    let mut total = 0f64;
    let mut count = 0u64;
    let mut local_box = (0i64, 0i64, width as i64, height as i64);
    let mut y = 0usize;
    while y < height {
        let box_y2 = std::cmp::min(height, y + cell_height);
        let mut x = 0usize;
        while x < width {
            let box_x2 = std::cmp::min(width, x + cell_width);
            let mut sum = 0f64;
            let mut cell_count = 0u64;
            for cell_y in y..box_y2 {
                let row = cell_y * width;
                for cell_x in x..box_x2 {
                    sum += diff[row + cell_x] as f64;
                    cell_count += 1;
                }
            }
            if cell_count > 0 {
                let value = sum / cell_count as f64;
                total += value;
                count += 1;
                if value > best {
                    best = value;
                    local_box = (x as i64, y as i64, box_x2 as i64, box_y2 as i64);
                }
            }
            x += cell_width;
        }
        y += cell_height;
    }
    let change_t_local_max = best;
    let change_t_local_mean = if count > 0 { total / count as f64 } else { 0.0 };

    let mut change_t_roi_max = 0f64;
    let mut target_box: Option<(i64, i64, i64, i64)> = None;
    for (raw_x1, raw_y1, raw_x2, raw_y2) in rois.iter().copied() {
        if raw_x2 <= raw_x1 || raw_y2 <= raw_y1 {
            continue;
        }
        let x1 = clamp_i64(raw_x1, 0, width as i64) as usize;
        let y1 = clamp_i64(raw_y1, 0, height as i64) as usize;
        let x2 = clamp_i64(raw_x2, 0, width as i64) as usize;
        let y2 = clamp_i64(raw_y2, 0, height as i64) as usize;
        let mut sum = 0f64;
        let mut roi_count = 0u64;
        for roi_y in y1..y2 {
            let row = roi_y * width;
            for roi_x in x1..x2 {
                sum += diff[row + roi_x] as f64;
                roi_count += 1;
            }
        }
        let roi_change = if roi_count > 0 {
            sum / roi_count as f64
        } else {
            0.0
        };
        if roi_change > change_t_roi_max {
            change_t_roi_max = roi_change;
            target_box = Some((raw_x1, raw_y1, raw_x2, raw_y2));
        }
    }

    let mut change_t_context_mean = change_t_local_mean;
    if let Some((x1, y1, x2, y2)) = target_box {
        let box_width = std::cmp::max(1, x2 - x1);
        let box_height = std::cmp::max(1, y2 - y1);
        let dx = (box_width as f64 * expand_margin) as i64;
        let dy = (box_height as f64 * expand_margin) as i64;
        let outer_x1 = std::cmp::max(0, x1.saturating_sub(dx));
        let outer_y1 = std::cmp::max(0, y1.saturating_sub(dy));
        let outer_x2 = std::cmp::min(width as i64, x2.saturating_add(dx));
        let outer_y2 = std::cmp::min(height as i64, y2.saturating_add(dy));
        if outer_x2 > outer_x1 && outer_y2 > outer_y1 {
            let mut sum = 0f64;
            let mut context_count = 0u64;
            for context_y in outer_y1..outer_y2 {
                let row = context_y as usize * width;
                let inside_y = context_y >= y1 && context_y < y2;
                for context_x in outer_x1..outer_x2 {
                    if inside_y && context_x >= x1 && context_x < x2 {
                        continue;
                    }
                    sum += diff[row + context_x as usize] as f64;
                    context_count += 1;
                }
            }
            if context_count > 0 {
                change_t_context_mean = sum / context_count as f64;
            }
        }
    }

    let target_box_object: PyObject = match target_box {
        Some(value) => value.into_py(py),
        None => py.None(),
    };
    Ok((
        change_t_global,
        change_t_local_max,
        change_t_local_mean,
        local_box,
        change_t_roi_max,
        target_box_object,
        change_t_context_mean,
    )
        .into_py(py))
}

fn compute_a3b_box_stats(
    edges: &numpy::ndarray::ArrayView2<'_, u8>,
    gray_view: &numpy::ndarray::ArrayView2<'_, u8>,
    x1: i64,
    y1: i64,
    x2: i64,
    y2: i64,
) -> PyResult<A3bBoxStats> {
    let height = edges.shape()[0];
    let width = edges.shape()[1];
    let (x1, y1, x2, y2) = validate_box(x1, y1, x2, y2, width, height)?;
    let box_width = x2 - x1;
    let box_height = y2 - y1;
    let area = box_width
        .checked_mul(box_height)
        .ok_or_else(|| PyValueError::new_err("bbox area is too large"))? as f64;

    let mut edge_sum = 0u64;
    let mut gray_sum = 0f64;
    for y in y1..y2 {
        for x in x1..x2 {
            edge_sum += unsafe { *edges.uget((y, x)) } as u64;
            gray_sum += unsafe { *gray_view.uget((y, x)) } as f64;
        }
    }
    let edge_density = edge_sum as f64 / area;
    let gray_mean = gray_sum / area;
    let mut variance_sum = 0f64;
    for y in y1..y2 {
        for x in x1..x2 {
            let delta = unsafe { *gray_view.uget((y, x)) } as f64 - gray_mean;
            variance_sum += delta * delta;
        }
    }
    let gray_std = (variance_sum / area).sqrt();

    let border = std::cmp::max(
        2i64,
        (std::cmp::min(box_width, box_height) as f64 * 0.035) as i64,
    );
    let half_height = std::cmp::max(1i64, (box_height / 2) as i64);
    let half_width = std::cmp::max(1i64, (box_width / 2) as i64);
    let ring_width = std::cmp::max(
        1i64,
        std::cmp::min(border, std::cmp::min(half_height, half_width)),
    ) as usize;

    let mut border_edge_sum = 0u64;
    let mut border_gray_sum = 0f64;
    let mut border_count = 0u64;
    for y in y1..(y1 + ring_width) {
        for x in x1..x2 {
            border_edge_sum += unsafe { *edges.uget((y, x)) } as u64;
            border_gray_sum += unsafe { *gray_view.uget((y, x)) } as f64;
            border_count += 1;
        }
    }
    for y in (y2 - ring_width)..y2 {
        for x in x1..x2 {
            border_edge_sum += unsafe { *edges.uget((y, x)) } as u64;
            border_gray_sum += unsafe { *gray_view.uget((y, x)) } as f64;
            border_count += 1;
        }
    }
    for y in y1..y2 {
        for x in x1..(x1 + ring_width) {
            border_edge_sum += unsafe { *edges.uget((y, x)) } as u64;
            border_gray_sum += unsafe { *gray_view.uget((y, x)) } as f64;
            border_count += 1;
        }
    }
    for y in y1..y2 {
        for x in (x2 - ring_width)..x2 {
            border_edge_sum += unsafe { *edges.uget((y, x)) } as u64;
            border_gray_sum += unsafe { *gray_view.uget((y, x)) } as f64;
            border_count += 1;
        }
    }
    let border_edge_density = border_edge_sum as f64 / border_count as f64;
    let border_mean = border_gray_sum / border_count as f64;

    let (inner_edge_density, inner_mean) =
        if box_width as i64 > 2 * border + 2 && box_height as i64 > 2 * border + 2 {
            let inner_border = border as usize;
            let inner_x1 = x1 + inner_border;
            let inner_x2 = x2 - inner_border;
            let inner_y1 = y1 + inner_border;
            let inner_y2 = y2 - inner_border;
            let inner_area = (inner_x2 - inner_x1) * (inner_y2 - inner_y1);
            let mut inner_edge_sum = 0u64;
            let mut inner_gray_sum = 0f64;
            for y in inner_y1..inner_y2 {
                for x in inner_x1..inner_x2 {
                    inner_edge_sum += unsafe { *edges.uget((y, x)) } as u64;
                    inner_gray_sum += unsafe { *gray_view.uget((y, x)) } as f64;
                }
            }
            (
                inner_edge_sum as f64 / inner_area as f64,
                inner_gray_sum / inner_area as f64,
            )
        } else {
            (edge_density, border_mean)
        };

    Ok((
        edge_density,
        border_edge_density,
        inner_edge_density,
        border_mean,
        inner_mean,
        gray_std,
    ))
}

/// Compute six A3b statistics for one validated media-candidate box.
#[pyfunction]
fn a3b_one_box_stats(
    py: Python<'_>,
    edge_mask: PyReadonlyArray2<u8>,
    gray: PyReadonlyArray2<u8>,
    x1: i64,
    y1: i64,
    x2: i64,
    y2: i64,
) -> PyResult<PyObject> {
    let edges = edge_mask.as_array();
    let gray_view = gray.as_array();
    validate_same_shape("edge_mask", edges.shape(), "gray", gray_view.shape())?;
    let edges_owned = edges.to_owned();
    let gray_owned = gray_view.to_owned();
    let result = py.allow_threads(move || {
        compute_a3b_box_stats(&edges_owned.view(), &gray_owned.view(), x1, y1, x2, y2)
    })?;
    Ok(result.into_py(py))
}

/// Compute A3b statistics for all media-candidate boxes in one Python call.
///
/// The output list preserves input order, and each item is exactly the
/// six-field tuple returned by `a3b_one_box_stats`. An empty box list returns
/// an empty list. Any invalid box rejects the full batch with `ValueError`.
#[pyfunction]
fn a3b_boxes_stats(
    py: Python<'_>,
    edge_mask: PyReadonlyArray2<u8>,
    gray: PyReadonlyArray2<u8>,
    boxes: Vec<(i64, i64, i64, i64)>,
) -> PyResult<PyObject> {
    let edges = edge_mask.as_array();
    let gray_view = gray.as_array();
    validate_same_shape("edge_mask", edges.shape(), "gray", gray_view.shape())?;
    let edges_owned = edges.to_owned();
    let gray_owned = gray_view.to_owned();
    let results = py.allow_threads(move || {
        let edges = edges_owned.view();
        let gray_view = gray_owned.view();
        let mut results = Vec::with_capacity(boxes.len());
        for (x1, y1, x2, y2) in boxes {
            results.push(compute_a3b_box_stats(&edges, &gray_view, x1, y1, x2, y2)?);
        }
        Ok::<_, PyErr>(results)
    })?;
    Ok(results.into_py(py))
}

/// Return `(maximum cell mean, mean of cell means, maximum cell bbox)`.
#[pyfunction]
fn best_grid_value_f32(
    py: Python<'_>,
    array: PyReadonlyArray2<f32>,
    grid: i64,
) -> PyResult<PyObject> {
    let values = array.as_array();
    let height = values.shape()[0];
    let width = values.shape()[1];
    let grid = std::cmp::max(1, grid) as usize;
    let cell_width = std::cmp::max(8, width / grid);
    let cell_height = std::cmp::max(8, height / grid);
    let mut best = 0f64;
    let mut total = 0f64;
    let mut count = 0u64;
    let mut best_box = (0i64, 0i64, width as i64, height as i64);
    let mut y = 0usize;
    while y < height {
        let box_y2 = std::cmp::min(height, y + cell_height);
        let mut x = 0usize;
        while x < width {
            let box_x2 = std::cmp::min(width, x + cell_width);
            let mut sum = 0f64;
            let mut cell_count = 0u64;
            for cell_y in y..box_y2 {
                for cell_x in x..box_x2 {
                    sum += unsafe { *values.uget((cell_y, cell_x)) } as f64;
                    cell_count += 1;
                }
            }
            if cell_count > 0 {
                let value = sum / cell_count as f64;
                total += value;
                count += 1;
                if value > best {
                    best = value;
                    best_box = (x as i64, y as i64, box_x2 as i64, box_y2 as i64);
                }
            }
            x += cell_width;
        }
        y += cell_height;
    }
    let mean = if count > 0 { total / count as f64 } else { 0.0 };
    Ok((best, mean, best_box).into_py(py))
}

#[inline(always)]
fn reflect101(index: i64, length: i64) -> usize {
    if length <= 1 {
        return 0;
    }
    let mut reflected = index;
    loop {
        if reflected < 0 {
            reflected = -reflected;
        } else if reflected >= length {
            reflected = 2 * (length - 1) - reflected;
        } else {
            break;
        }
    }
    reflected as usize
}

/// Equivalent to `cv2.Laplacian(gray, cv2.CV_32F, ksize=1).var()`.
#[pyfunction]
fn blinding_laplacian_var(gray: PyReadonlyArray2<u8>) -> PyResult<f64> {
    let values = gray.as_array();
    let height = values.shape()[0] as i64;
    let width = values.shape()[1] as i64;
    if height == 0 || width == 0 {
        return Ok(0.0);
    }
    let mut sum = 0f64;
    let mut squared_sum = 0f64;
    for y in 0..height {
        let y_index = y as usize;
        let up_y = reflect101(y - 1, height);
        let down_y = reflect101(y + 1, height);
        for x in 0..width {
            let x_index = x as usize;
            let left_x = reflect101(x - 1, width);
            let right_x = reflect101(x + 1, width);
            let center = unsafe { *values.uget((y_index, x_index)) } as f32;
            let up = unsafe { *values.uget((up_y, x_index)) } as f32;
            let down = unsafe { *values.uget((down_y, x_index)) } as f32;
            let left = unsafe { *values.uget((y_index, left_x)) } as f32;
            let right = unsafe { *values.uget((y_index, right_x)) } as f32;
            let laplacian = up + down + left + right - 4.0f32 * center;
            let laplacian_f64 = laplacian as f64;
            sum += laplacian_f64;
            squared_sum += laplacian_f64 * laplacian_f64;
        }
    }
    let count = (height * width) as f64;
    let mean = sum / count;
    let variance = squared_sum / count - mean * mean;
    Ok(variance.max(0.0))
}

#[pyfunction]
fn api_version() -> &'static str {
    API_VERSION
}

#[pyfunction]
fn a3b_batch_api_version() -> &'static str {
    A3B_BATCH_API_VERSION
}

#[pyfunction]
fn crate_version() -> &'static str {
    CRATE_VERSION
}

#[pyfunction]
fn capabilities() -> Vec<&'static str> {
    CAPABILITIES.to_vec()
}

#[pyfunction]
fn build_info() -> Vec<(&'static str, &'static str)> {
    vec![
        ("crate_name", env!("CARGO_PKG_NAME")),
        ("crate_version", CRATE_VERSION),
        ("api_version", API_VERSION),
        ("a3b_batch_api_version", A3B_BATCH_API_VERSION),
        ("target_os", std::env::consts::OS),
        ("target_arch", std::env::consts::ARCH),
        ("target_triple", env!("MODULE_A_NATIVE_BUILD_TARGET")),
        ("profile", env!("MODULE_A_NATIVE_BUILD_PROFILE")),
        ("rustc_version", env!("MODULE_A_NATIVE_RUSTC_VERSION")),
        ("panic_strategy", "unwind"),
        ("source_manifest_sha256", SOURCE_MANIFEST_SHA256),
    ]
}

#[pymodule]
fn module_a_native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", CRATE_VERSION)?;
    module.add("API_VERSION", API_VERSION)?;
    module.add("A3B_BATCH_API_VERSION", A3B_BATCH_API_VERSION)?;
    module.add_function(wrap_pyfunction!(a1_lbp_features, module)?)?;
    module.add_function(wrap_pyfunction!(a2_change_features, module)?)?;
    module.add_function(wrap_pyfunction!(a3b_boxes_stats, module)?)?;
    module.add_function(wrap_pyfunction!(a3b_one_box_stats, module)?)?;
    module.add_function(wrap_pyfunction!(best_grid_value_f32, module)?)?;
    module.add_function(wrap_pyfunction!(blinding_laplacian_var, module)?)?;
    module.add_function(wrap_pyfunction!(api_version, module)?)?;
    module.add_function(wrap_pyfunction!(a3b_batch_api_version, module)?)?;
    module.add_function(wrap_pyfunction!(crate_version, module)?)?;
    module.add_function(wrap_pyfunction!(capabilities, module)?)?;
    module.add_function(wrap_pyfunction!(build_info, module)?)?;
    Ok(())
}
