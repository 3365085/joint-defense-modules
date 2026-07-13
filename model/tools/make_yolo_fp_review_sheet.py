from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成YOLO安全帽误检复核拼图。")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("materials/yolo_fp_review/review_sheets"))
    parser.add_argument("--limit", type=int, default=48)
    parser.add_argument("--thumb-width", type=int, default=320)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        return list(csv.DictReader(fp))


def make_sheet(rows: list[dict[str, str]], thumb_width: int) -> np.ndarray:
    thumbs: list[np.ndarray] = []
    for row in rows:
        image = cv2.imread(row["frame_path"])
        if image is None:
            continue
        height, width = image.shape[:2]
        thumb_height = max(1, int(height * thumb_width / max(1, width)))
        thumb = cv2.resize(image, (thumb_width, thumb_height))
        header = np.zeros((48, thumb_width, 3), dtype=np.uint8)
        label = f"{Path(row['frame_path']).name} | {row.get('category', '')} | {row.get('bucket', '')}"
        cv2.putText(header, label[:90], (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1)
        thumbs.append(np.vstack([header, thumb]))
    if not thumbs:
        return np.zeros((32, 32, 3), dtype=np.uint8)
    cols = 3
    rows_count = int(np.ceil(len(thumbs) / cols))
    cell_height = max(thumb.shape[0] for thumb in thumbs)
    blank = np.zeros((cell_height, thumb_width, 3), dtype=np.uint8)
    canvas_rows: list[np.ndarray] = []
    idx = 0
    for _ in range(rows_count):
        row_cells = []
        for _ in range(cols):
            if idx < len(thumbs):
                thumb = thumbs[idx]
                if thumb.shape[0] < cell_height:
                    pad = np.zeros((cell_height - thumb.shape[0], thumb_width, 3), dtype=np.uint8)
                    thumb = np.vstack([thumb, pad])
                row_cells.append(thumb)
            else:
                row_cells.append(blank.copy())
            idx += 1
        canvas_rows.append(np.hstack(row_cells))
    return np.vstack(canvas_rows)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.manifest)[: max(0, args.limit)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise SystemExit("manifest中没有可用记录")
    priority = {"head_helmet_overlap": 0, "helmet_fp": 1, "uncertain": 2, "uncategorized": 3}
    unique_rows: dict[str, dict[str, str]] = {}
    for row in rows:
        frame_path = row.get("frame_path", "")
        if not frame_path:
            continue
        current = unique_rows.get(frame_path)
        if current is None or priority.get(row.get("bucket", "uncategorized"), 99) < priority.get(current.get("bucket", "uncategorized"), 99):
            unique_rows[frame_path] = row
    by_bucket: dict[str, list[dict[str, str]]] = {}
    for row in unique_rows.values():
        bucket = row.get("bucket", "").strip()
        if not bucket:
            continue
        by_bucket.setdefault(bucket, []).append(row)
    if not by_bucket:
        raise SystemExit("manifest中没有候选bucket记录")
    for bucket, bucket_rows in by_bucket.items():
        sheet = make_sheet(bucket_rows, args.thumb_width)
        output_path = args.output_dir / f"{bucket}_contact_sheet.jpg"
        cv2.imwrite(str(output_path), sheet)
        print(output_path)


if __name__ == "__main__":
    main()
