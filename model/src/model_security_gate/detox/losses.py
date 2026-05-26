from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from model_security_gate.detox.feature_hooks import attention_map, bbox_union_mask_from_batch


def _zero_like_model_loss(model: torch.nn.Module) -> torch.Tensor:
    p = next(model.parameters(), None)
    if p is None:
        return torch.tensor(0.0)
    return p.sum() * 0.0


def _ensure_ultralytics_loss_hyp(model: torch.nn.Module) -> None:
    """Make exported Ultralytics models usable with DetectionModel.loss().

    Some `.pt` files restore `model.args` as a plain dict containing only a few
    training keys. Ultralytics' loss path expects attribute-style hyperparameters
    such as `hyp.box`, `hyp.cls`, and `hyp.dfl`. Without this guard, custom detox
    training crashes before doing any useful work.
    """
    defaults = {
        "box": 7.5,
        "cls": 0.5,
        "dfl": 1.5,
        "pose": 12.0,
        "kobj": 1.0,
        "label_smoothing": 0.0,
    }
    args = getattr(model, "args", None)
    if isinstance(args, Mapping):
        data = dict(defaults)
        data.update(dict(args))
        model.args = SimpleNamespace(**data)
    elif args is None or not all(hasattr(args, key) for key in ("box", "cls", "dfl")):
        data = dict(defaults)
        if args is not None and hasattr(args, "__dict__"):
            data.update(vars(args))
        model.args = SimpleNamespace(**data)

    criterion = getattr(model, "criterion", None)
    if criterion is None and hasattr(model, "init_criterion"):
        try:
            criterion = model.init_criterion()
            model.criterion = criterion
        except Exception:
            criterion = getattr(model, "criterion", None)
    hyp = getattr(criterion, "hyp", None)
    if isinstance(hyp, Mapping):
        data = dict(defaults)
        data.update(dict(hyp))
        criterion.hyp = SimpleNamespace(**data)
    if criterion is not None:
        try:
            device = next(model.parameters()).device
            if hasattr(criterion, "proj") and torch.is_tensor(criterion.proj):
                criterion.proj = criterion.proj.to(device)
        except StopIteration:
            pass


def _looks_like_yolov5_official(model: torch.nn.Module) -> bool:
    module_name = type(model).__module__.lower()
    return module_name.startswith("models.") and hasattr(model, "names")


def _decoded_prediction_bnc(pred: torch.Tensor) -> torch.Tensor:
    if pred.ndim != 3:
        raise ValueError(f"Expected 3D prediction tensor, got {tuple(pred.shape)}")
    if pred.shape[2] <= 16 and pred.shape[1] > pred.shape[2]:
        return pred
    if pred.shape[1] <= 16 and pred.shape[2] > pred.shape[1]:
        return pred.transpose(1, 2).contiguous()
    return pred if pred.shape[2] <= pred.shape[1] else pred.transpose(1, 2).contiguous()


def _label_xyxy_pixels(boxes_xywhn: torch.Tensor, img_w: float, img_h: float) -> torch.Tensor:
    if boxes_xywhn.numel() == 0:
        return torch.zeros((0, 4), device=boxes_xywhn.device, dtype=boxes_xywhn.dtype)
    xy = boxes_xywhn[:, :2] * boxes_xywhn.new_tensor([float(img_w), float(img_h)])
    wh = boxes_xywhn[:, 2:].clamp_min(0.0) * boxes_xywhn.new_tensor([float(img_w), float(img_h)])
    x1y1 = xy - wh / 2.0
    x2y2 = xy + wh / 2.0
    return torch.cat([x1y1, x2y2], dim=1)


def _class_near_mask(
    pred_xywh: torch.Tensor,
    pred_xyxy: torch.Tensor,
    gt_boxes_xywhn: torch.Tensor,
    img_w: float,
    img_h: float,
    *,
    iou_threshold: float = 0.05,
    center_radius: float = 1.75,
) -> torch.Tensor:
    if gt_boxes_xywhn.numel() == 0:
        return torch.zeros((pred_xywh.shape[0],), device=pred_xywh.device, dtype=torch.bool)
    gt_xyxy = _label_xyxy_pixels(gt_boxes_xywhn.to(pred_xywh.device), img_w, img_h)
    ious = _box_iou_xyxy(pred_xyxy, gt_xyxy)
    near_iou = ious.max(dim=1).values >= float(iou_threshold)
    gt_centers = gt_boxes_xywhn[:, :2].to(pred_xywh.device) * gt_boxes_xywhn.new_tensor([float(img_w), float(img_h)]).to(pred_xywh.device)
    gt_wh = gt_boxes_xywhn[:, 2:].clamp_min(1e-6).to(pred_xywh.device) * gt_boxes_xywhn.new_tensor([float(img_w), float(img_h)]).to(pred_xywh.device)
    pred_centers = pred_xywh[:, :2]
    dx = (pred_centers[:, None, 0] - gt_centers[None, :, 0]).abs() / (gt_wh[None, :, 0] / 2.0 * float(center_radius)).clamp_min(1.0)
    dy = (pred_centers[:, None, 1] - gt_centers[None, :, 1]).abs() / (gt_wh[None, :, 1] / 2.0 * float(center_radius)).clamp_min(1.0)
    near_center = ((dx <= 1.0) & (dy <= 1.0)).any(dim=1)
    return near_iou | near_center


def _yolov5_official_proxy_loss(model: torch.nn.Module, batch: Dict[str, Any]) -> torch.Tensor:
    """Differentiable repair proxy for bundled YOLOv5 checkpoints.

    The first version only rewarded target-class presence near labels. That was
    too weak: a model could reduce the loss by emitting many high-confidence
    helmet/head boxes everywhere. This proxy also suppresses class-specific
    false positives away from same-class labels, so repair training must improve
    recall without drifting into detection spam.
    """
    was_training = model.training
    model.eval()
    out = model(batch["img"])
    decoded = _find_decoded_prediction(out)
    if decoded is None:
        model.train(was_training)
        return _zero_like_model_loss(model)
    pred = _decoded_prediction_bnc(decoded)
    if pred.shape[-1] < 6:
        model.train(was_training)
        return _zero_like_model_loss(model)

    imgs = batch["img"]
    _, _, img_h, img_w = imgs.shape
    xywh = pred[..., :4]
    obj = pred[..., 4].clamp(1e-6, 1.0)
    cls_scores = pred[..., 5:].clamp(1e-6, 1.0)
    batch_idx = batch.get("batch_idx", torch.zeros((0,), device=imgs.device)).long().view(-1)
    classes = batch.get("cls", torch.zeros((0, 1), device=imgs.device)).long().view(-1)
    boxes = batch.get("bboxes", torch.zeros((0, 4), device=imgs.device)).float().reshape(-1, 4)

    losses: List[torch.Tensor] = []
    eps = torch.tensor(1e-6, device=imgs.device)
    nc = int(cls_scores.shape[-1])
    for i in range(int(boxes.shape[0])):
        b = int(batch_idx[i].item()) if i < int(batch_idx.numel()) else 0
        c = int(classes[i].item()) if i < int(classes.numel()) else -1
        if b < 0 or b >= pred.shape[0] or c < 0 or c >= cls_scores.shape[-1]:
            continue
        box = boxes[i]
        cx = box[0] * float(img_w)
        cy = box[1] * float(img_h)
        bw = box[2].clamp_min(1e-3) * float(img_w)
        bh = box[3].clamp_min(1e-3) * float(img_h)
        px = xywh[b, :, 0]
        py = xywh[b, :, 1]
        sx = torch.clamp(bw * 1.5, min=16.0)
        sy = torch.clamp(bh * 1.5, min=16.0)
        spatial = torch.exp(-0.5 * (((px - cx) / sx) ** 2 + ((py - cy) / sy) ** 2))
        pred_xyxy = _xywh_to_xyxy_pixels(xywh[b], img_w=float(img_w), img_h=float(img_h))
        gt_xyxy = _label_xyxy_pixels(box.view(1, 4), float(img_w), float(img_h))
        iou_quality = _box_iou_xyxy(pred_xyxy, gt_xyxy).view(-1).clamp_min(0.0)
        localization = spatial * (0.25 + 0.75 * iou_quality)
        score = (obj[b] * cls_scores[b, :, c] * localization).max()
        losses.append(-torch.log(score.clamp_min(eps)))

    fp_losses: List[torch.Tensor] = []
    fp_margin = torch.tensor(0.20, device=imgs.device, dtype=pred.dtype)
    fp_topk = 64
    fp_weight = 0.35
    # PPE repair currently targets helmet/head. If a model has fewer classes,
    # fall back gracefully to the available class channels.
    suppress_class_ids = [cid for cid in (0, 1) if cid < nc]
    for b in range(int(pred.shape[0])):
        image_sel = batch_idx == b
        image_classes = classes[image_sel] if classes.numel() else classes
        image_boxes = boxes[image_sel] if boxes.numel() else boxes.reshape(0, 4)
        pred_xywh = xywh[b]
        pred_xyxy = _xywh_to_xyxy_pixels(pred_xywh, img_w=float(img_w), img_h=float(img_h))
        for cid in suppress_class_ids:
            class_boxes = image_boxes[image_classes == int(cid)] if image_classes.numel() else image_boxes.reshape(0, 4)
            near_same_class = _class_near_mask(pred_xywh, pred_xyxy, class_boxes, float(img_w), float(img_h))
            fp_scores = (obj[b] * cls_scores[b, :, int(cid)])[~near_same_class]
            if fp_scores.numel() == 0:
                continue
            k = min(fp_topk, int(fp_scores.numel()))
            hard_scores = torch.topk(fp_scores, k=k).values
            fp_losses.append(F.relu(hard_scores - fp_margin).pow(2).mean())
    model.train(was_training)
    if fp_losses:
        losses.append(torch.stack(fp_losses).mean() * fp_weight)
    if not losses:
        return _zero_like_model_loss(model)
    return torch.stack(losses).mean()


def _yolov5_training_predictions(obj: Any) -> list[torch.Tensor]:
    if isinstance(obj, list) and obj and all(torch.is_tensor(x) and x.ndim == 5 for x in obj):
        return [x for x in obj]
    if isinstance(obj, tuple):
        for item in obj:
            found = _yolov5_training_predictions(item)
            if found:
                return found
    return []


def _yolov5_targets_from_batch(batch: Dict[str, Any]) -> torch.Tensor:
    imgs = batch["img"]
    device = imgs.device
    batch_idx = batch.get("batch_idx", torch.zeros((0,), device=device)).float().view(-1, 1)
    classes = batch.get("cls", torch.zeros((0, 1), device=device)).float().view(-1, 1)
    boxes = batch.get("bboxes", torch.zeros((0, 4), device=device)).float().reshape(-1, 4)
    if batch_idx.numel() == 0 or boxes.numel() == 0:
        return torch.zeros((0, 6), device=device, dtype=imgs.dtype)
    return torch.cat((batch_idx.to(device), classes.to(device), boxes.to(device)), dim=1).to(dtype=imgs.dtype)


def _yolov5_build_targets(
    pred: Sequence[torch.Tensor],
    targets: torch.Tensor,
    detect: Any,
    *,
    anchor_t: float = 4.0,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]], list[torch.Tensor]]:
    na = int(getattr(detect, "na", 0) or 0)
    nt = int(targets.shape[0])
    device = targets.device
    tcls: list[torch.Tensor] = []
    tbox: list[torch.Tensor] = []
    indices: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    anch: list[torch.Tensor] = []
    if na <= 0:
        return tcls, tbox, indices, anch

    gain = torch.ones(7, device=device, dtype=targets.dtype)
    ai = torch.arange(na, device=device, dtype=targets.dtype).view(na, 1).repeat(1, nt)
    targets_ai = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), dim=2) if nt else torch.zeros((na, 0, 7), device=device, dtype=targets.dtype)
    g = 0.5
    off = torch.tensor(
        [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]],
        device=device,
        dtype=targets.dtype,
    ) * g

    anchors_all = getattr(detect, "anchors", None)
    for i, pi in enumerate(pred):
        anchors = anchors_all[i].to(device=device, dtype=targets.dtype)
        _, _, ny, nx, _ = pi.shape
        gain[2:6] = torch.tensor([nx, ny, nx, ny], device=device, dtype=targets.dtype)
        t = targets_ai * gain
        if nt:
            r = t[..., 4:6] / anchors[:, None]
            match = torch.max(r, 1.0 / r).max(2).values < float(anchor_t)
            t = t[match]
            if t.numel():
                gxy = t[:, 2:4]
                gxi = gain[[2, 3]] - gxy
                j, k = ((gxy % 1.0 < g) & (gxy > 1.0)).T
                l, m = ((gxi % 1.0 < g) & (gxi > 1.0)).T
                keep = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[keep]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[keep]
            else:
                offsets = torch.zeros((0, 2), device=device, dtype=targets.dtype)
        else:
            t = targets_ai[0]
            offsets = torch.zeros((0, 2), device=device, dtype=targets.dtype)

        if t.numel():
            bc = t[:, :2].long()
            gxy = t[:, 2:4]
            gwh = t[:, 4:6]
            a = t[:, 6].long()
            gij = (gxy - offsets).long()
            gi, gj = gij.T
            b, c = bc.T
            indices.append((b, a, gj.clamp(0, ny - 1), gi.clamp(0, nx - 1)))
            tbox.append(torch.cat((gxy - gij, gwh), dim=1))
            anch.append(anchors[a])
            tcls.append(c)
        else:
            empty_long = torch.zeros((0,), device=device, dtype=torch.long)
            indices.append((empty_long, empty_long, empty_long, empty_long))
            tbox.append(torch.zeros((0, 4), device=device, dtype=targets.dtype))
            anch.append(torch.zeros((0, 2), device=device, dtype=targets.dtype))
            tcls.append(empty_long)
    return tcls, tbox, indices, anch


def _yolov5_official_compute_loss(model: torch.nn.Module, batch: Dict[str, Any]) -> torch.Tensor:
    """Minimal YOLOv5 detection loss for the bundled official checkpoint."""
    from utils.metrics import bbox_iou

    was_training = model.training
    model.train()
    pred = _yolov5_training_predictions(model(batch["img"]))
    if not pred:
        model.train(was_training)
        return _yolov5_official_proxy_loss(model, batch)
    detect = getattr(model, "model", [None])[-1]
    nc = int(getattr(detect, "nc", max(1, pred[0].shape[-1] - 5)))
    device = batch["img"].device
    dtype = pred[0].dtype
    targets = _yolov5_targets_from_batch(batch)
    tcls, tbox, indices, anchors = _yolov5_build_targets(pred, targets, detect)

    bce_cls = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=device, dtype=dtype))
    bce_obj = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=device, dtype=dtype))
    balance = [4.0, 1.0, 0.4, 0.1]
    lcls = torch.zeros((), device=device, dtype=dtype)
    lbox = torch.zeros((), device=device, dtype=dtype)
    lobj = torch.zeros((), device=device, dtype=dtype)
    target_obj_ratio = 1.0
    cp, cn = 1.0, 0.0

    for i, pi in enumerate(pred):
        b, a, gj, gi = indices[i]
        tobj = torch.zeros(pi.shape[:4], device=device, dtype=dtype)
        n = int(b.shape[0])
        if n:
            pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, nc), dim=1)
            pxy = pxy.sigmoid() * 2.0 - 0.5
            pwh = (pwh.sigmoid() * 2.0).pow(2) * anchors[i]
            pbox = torch.cat((pxy, pwh), dim=1)
            iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze().clamp(-1.0, 1.0)
            lbox = lbox + (1.0 - iou).mean()
            tobj[b, a, gj, gi] = (1.0 - target_obj_ratio) + target_obj_ratio * iou.detach().clamp(0.0).to(dtype)
            if nc > 1:
                target_cls = torch.full_like(pcls, cn)
                target_cls[torch.arange(n, device=device), tcls[i].clamp(0, nc - 1)] = cp
                lcls = lcls + bce_cls(pcls, target_cls)
        lobj = lobj + bce_obj(pi[..., 4], tobj) * float(balance[i] if i < len(balance) else balance[-1])

    # Official YOLOv5 defaults are box=0.05, cls=0.5, obj=1.0. Keep the same
    # ratios while returning a mean-like scalar for stable repair learning rates.
    loss = lbox * 0.05 + lobj * 1.0 + lcls * 0.5
    model.train(was_training)
    return loss


def yolov5_official_loss(model: torch.nn.Module, batch: Dict[str, Any], *, mode: str = "proxy") -> torch.Tensor:
    mode = str(mode or "proxy").strip().lower()
    if mode in {"compute", "official", "yolov5", "raw"}:
        return _yolov5_official_compute_loss(model, batch)
    if mode in {"combined", "hybrid"}:
        return _yolov5_official_compute_loss(model, batch) + _yolov5_official_proxy_loss(model, batch) * 0.25
    return _yolov5_official_proxy_loss(model, batch)


def supervised_yolo_loss(model: torch.nn.Module, batch: Dict[str, Any]) -> torch.Tensor:
    """Return Ultralytics DetectionModel supervised loss from a batch dict.

    Ultralytics DetectionModel.forward(batch_dict) returns either a scalar loss
    or a tuple whose first element is the scalar loss. This wrapper normalizes
    those variants and keeps a safe fallback for dry runs.
    """
    if _looks_like_yolov5_official(model):
        return yolov5_official_loss(model, batch, mode="proxy")
    _ensure_ultralytics_loss_hyp(model)
    try:
        out = model(batch)
    except TypeError:
        return _yolov5_official_proxy_loss(model, batch)
    if torch.is_tensor(out):
        return out.mean()
    if isinstance(out, (tuple, list)) and out:
        first = out[0]
        if torch.is_tensor(first):
            return first.mean()
        tensors = [x.mean() for x in out if torch.is_tensor(x)]
        if tensors:
            return torch.stack(tensors).sum()
    if isinstance(out, dict):
        tensors = [v.mean() for v in out.values() if torch.is_tensor(v)]
        if tensors:
            return torch.stack(tensors).sum()
    return _zero_like_model_loss(model)


def yolov5_decoded_distillation_loss(
    student_prediction: Any,
    teacher_prediction: Any,
    *,
    max_candidates: int = 2048,
) -> torch.Tensor:
    """Keep a repaired YOLOv5 checkpoint close to the source detector.

    The repair loss intentionally changes target behavior on poison/counterfactual
    images. A small distillation term prevents broad distribution drift, especially
    class-specific false positives that were already acceptable in the source model.
    """
    student = _find_decoded_prediction(student_prediction)
    teacher = _find_decoded_prediction(teacher_prediction)
    if student is None or teacher is None:
        ref = student if torch.is_tensor(student) else teacher
        return ref.sum() * 0.0 if torch.is_tensor(ref) else torch.tensor(0.0)
    student = _decoded_prediction_bnc(student).float()
    teacher = _decoded_prediction_bnc(teacher).float().detach()
    if student.shape != teacher.shape or student.shape[-1] < 6:
        return student.sum() * 0.0
    score_s = student[..., 4:].amax(dim=-1)
    score_t = teacher[..., 4:].amax(dim=-1)
    score = torch.maximum(score_s.detach(), score_t)
    k = min(max(1, int(max_candidates)), int(score.shape[-1]))
    idx = torch.topk(score, k=k, dim=-1).indices
    gather_idx = idx.unsqueeze(-1).expand(-1, -1, student.shape[-1])
    student_hard = torch.gather(student, dim=1, index=gather_idx)
    teacher_hard = torch.gather(teacher, dim=1, index=gather_idx)
    # Confidence/class channels are the key drift surface for false positives.
    return F.smooth_l1_loss(student_hard[..., 4:], teacher_hard[..., 4:])


def clone_batch_with_img(batch: Dict[str, Any], img: torch.Tensor) -> Dict[str, Any]:
    out = dict(batch)
    out["img"] = img
    return out


def pgd_adversarial_images(
    model: torch.nn.Module,
    batch: Dict[str, Any],
    eps: float = 4.0 / 255.0,
    alpha: Optional[float] = None,
    steps: int = 2,
    random_start: bool = True,
) -> torch.Tensor:
    """I-BAU-style inner maximization on the supervised detection loss.

    This does not need to know the true trigger. It asks: what small input
    perturbation most destabilizes the current detector on clean/counterfactual
    labels? The outer detox step then trains the model to resist that shift.
    """
    was_training = model.training
    model.eval()
    x0 = batch["img"].detach()
    if alpha is None:
        alpha = eps / max(1, steps) * 1.5
    if random_start:
        delta = torch.empty_like(x0).uniform_(-eps, eps)
    else:
        delta = torch.zeros_like(x0)
    adv = (x0 + delta).clamp(0.0, 1.0).detach()
    for _ in range(max(1, int(steps))):
        adv.requires_grad_(True)
        adv_batch = clone_batch_with_img(batch, adv)
        loss = supervised_yolo_loss(model, adv_batch)
        grad = torch.autograd.grad(loss, adv, retain_graph=False, create_graph=False, allow_unused=True)[0]
        if grad is None:
            break
        adv = adv.detach() + float(alpha) * grad.sign()
        adv = torch.max(torch.min(adv, x0 + eps), x0 - eps).clamp(0.0, 1.0).detach()
    model.train(was_training)
    return adv.detach()


def attention_localization_loss(
    features: Dict[str, torch.Tensor],
    batch: Dict[str, Any],
    target_class_ids: Sequence[int],
    outside_weight: float = 1.0,
) -> torch.Tensor:
    """Encourage target-class evidence to live inside target boxes.

    For images that contain target labels, penalize attention mass outside the
    union of target boxes. For images without target labels, no penalty is added
    here; target suppression is handled by normal detection labels and target
    removal counterfactual samples.
    """
    losses: List[torch.Tensor] = []
    if not features:
        return torch.tensor(0.0, device=batch["img"].device)
    for _name, feat in features.items():
        if feat.ndim != 4:
            continue
        attn = attention_map(feat)
        b, _c, h, w = attn.shape
        for i in range(b):
            mask = bbox_union_mask_from_batch(batch, i, (h, w), class_ids=target_class_ids, device=attn.device)
            if mask.sum() <= 0:
                continue
            inside = (attn[i] * mask).sum()
            total = attn[i].sum().clamp_min(1e-6)
            outside = 1.0 - inside / total
            losses.append(outside * float(outside_weight))
    if not losses:
        return torch.tensor(0.0, device=batch["img"].device)
    return torch.stack(losses).mean()


def consistency_loss_between_outputs(out_a: Any, out_b: Any) -> torch.Tensor:
    from model_security_gate.detox.feature_hooks import output_distillation_loss

    return output_distillation_loss(out_a, out_b, mode="smooth_l1")


def raw_prediction(model: torch.nn.Module, img: torch.Tensor) -> Any:
    return model(img)


def _find_decoded_prediction(obj: Any) -> Optional[torch.Tensor]:
    """Return a decoded YOLO prediction tensor shaped BCH or BHC.

    Ultralytics YOLOv8/YOLO11 detection models commonly return a tuple whose
    first tensor is ``(B, 4 + nc, N)`` during inference-style forwards. Some
    versions expose ``(B, N, 4 + nc)``. This helper avoids depending on one exact
    wrapper version while keeping the ODA loss optional and safe.
    """
    if torch.is_tensor(obj) and obj.ndim == 3:
        if obj.shape[1] >= 5 or obj.shape[2] >= 5:
            return obj
        return None
    if isinstance(obj, (list, tuple)):
        for item in obj:
            found = _find_decoded_prediction(item)
            if found is not None:
                return found
    if isinstance(obj, dict):
        for item in obj.values():
            found = _find_decoded_prediction(item)
            if found is not None:
                return found
    return None


def _prediction_channels_first(pred: torch.Tensor) -> torch.Tensor:
    if pred.ndim != 3:
        raise ValueError(f"Expected 3D prediction tensor, got {tuple(pred.shape)}")
    # Prefer the compact channel dimension as C. For YOLO detection C is usually
    # 4 + number_of_classes and N is thousands of candidates.
    if pred.shape[1] >= 5 and pred.shape[2] < 5:
        return pred
    if pred.shape[2] >= 5 and pred.shape[1] < 5:
        return pred.transpose(1, 2).contiguous()
    if pred.shape[1] <= pred.shape[2]:
        return pred
    return pred.transpose(1, 2).contiguous()


def _xywh_to_xyxy_pixels(boxes_xywh: torch.Tensor, img_w: float, img_h: float) -> torch.Tensor:
    xy = boxes_xywh[:, :2]
    wh = boxes_xywh[:, 2:].clamp_min(0.0)
    x1y1 = xy - wh / 2.0
    x2y2 = xy + wh / 2.0
    out = torch.cat([x1y1, x2y2], dim=1)
    out[:, [0, 2]] = out[:, [0, 2]].clamp(0.0, float(img_w))
    out[:, [1, 3]] = out[:, [1, 3]].clamp(0.0, float(img_h))
    return out


def _box_iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.numel() == 0 or b.numel() == 0:
        return torch.zeros((a.shape[0], b.shape[0]), device=a.device)
    lt = torch.maximum(a[:, None, :2], b[None, :, :2])
    rb = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp_min(0.0)
    inter = wh[..., 0] * wh[..., 1]
    area_a = ((a[:, 2] - a[:, 0]).clamp_min(0.0) * (a[:, 3] - a[:, 1]).clamp_min(0.0))[:, None]
    area_b = ((b[:, 2] - b[:, 0]).clamp_min(0.0) * (b[:, 3] - b[:, 1]).clamp_min(0.0))[None, :]
    return inter / (area_a + area_b - inter).clamp_min(1e-6)


def target_recall_confidence_loss(
    prediction: Any,
    batch: Dict[str, Any],
    target_class_ids: Sequence[int],
    min_conf: float = 0.45,
    iou_threshold: float = 0.05,
    center_radius: float = 1.50,
    topk: int = 24,
    loss_scale: float = 1.0,
) -> torch.Tensor:
    """ODA recall-preserving loss for target-present images.

    ODA/backdoor disappearance failures usually keep the image visually valid
    while suppressing the target class near a real object. Supervised YOLO loss
    alone can be too diffuse in a mixed detox batch, so this loss adds a direct
    constraint: for every ground-truth target box, at least one decoded candidate
    near that box must keep target confidence above ``min_conf``.

    The matching mask is intentionally broad (IoU OR center-inside an expanded
    GT region), then the loss only backpropagates through class confidence. This
    makes it stable across YOLO versions and avoids fighting the box regressor.
    """
    pred = _find_decoded_prediction(prediction)
    if pred is None:
        ref = batch.get("img")
        return ref.sum() * 0.0 if torch.is_tensor(ref) else torch.tensor(0.0)
    pred = _prediction_channels_first(pred).float()
    if pred.shape[1] < 5 or not target_class_ids:
        return pred.sum() * 0.0

    device = pred.device
    cls = batch.get("cls", torch.zeros((0, 1), device=device)).view(-1).long().to(device)
    bboxes = batch.get("bboxes", torch.zeros((0, 4), device=device)).float().to(device)
    bidx = batch.get("batch_idx", torch.zeros((0,), device=device)).long().to(device)
    if cls.numel() == 0 or bboxes.numel() == 0:
        return pred.sum() * 0.0

    target_ids = torch.tensor([int(x) for x in target_class_ids], device=device, dtype=torch.long)
    target_sel = (cls[:, None] == target_ids[None, :]).any(dim=1)
    if not bool(target_sel.any()):
        return pred.sum() * 0.0

    img_h = float(batch["img"].shape[-2])
    img_w = float(batch["img"].shape[-1])
    losses: List[torch.Tensor] = []
    min_conf_t = torch.tensor(float(min_conf), device=device, dtype=pred.dtype)
    topk = max(1, int(topk))
    nc = pred.shape[1] - 4

    for label_index in torch.where(target_sel)[0].tolist():
        image_index = int(bidx[label_index].item())
        cid = int(cls[label_index].item())
        class_channel = 4 + cid
        if image_index < 0 or image_index >= pred.shape[0] or cid < 0 or cid >= nc:
            continue

        gt_xywhn = bboxes[label_index]
        gt_xc = gt_xywhn[0] * img_w
        gt_yc = gt_xywhn[1] * img_h
        gt_w = gt_xywhn[2].clamp_min(1e-6) * img_w
        gt_h = gt_xywhn[3].clamp_min(1e-6) * img_h
        gt_xyxy = torch.stack(
            [
                gt_xc - gt_w / 2.0,
                gt_yc - gt_h / 2.0,
                gt_xc + gt_w / 2.0,
                gt_yc + gt_h / 2.0,
            ]
        ).view(1, 4)

        image_pred = pred[image_index]
        pred_xywh = image_pred[:4].transpose(0, 1)
        pred_xyxy = _xywh_to_xyxy_pixels(pred_xywh, img_w=img_w, img_h=img_h)
        centers = pred_xywh[:, :2]
        dx = (centers[:, 0] - gt_xc).abs() / (gt_w / 2.0 * float(center_radius)).clamp_min(1.0)
        dy = (centers[:, 1] - gt_yc).abs() / (gt_h / 2.0 * float(center_radius)).clamp_min(1.0)
        center_match = (dx <= 1.0) & (dy <= 1.0)
        ious = _box_iou_xyxy(pred_xyxy, gt_xyxy).view(-1)
        near = center_match | (ious >= float(iou_threshold))

        scores = image_pred[class_channel]
        if bool(near.any()):
            candidate_scores = scores[near]
        else:
            # Fall back to nearest decoded candidates so tiny boxes or unusual
            # strides still contribute a useful gradient.
            dist = dx.square() + dy.square()
            k = min(topk, int(dist.numel()))
            candidate_scores = scores[torch.topk(-dist, k=k).indices]
        best_score = candidate_scores.max()
        losses.append(F.relu(min_conf_t - best_score).pow(2) * float(loss_scale))

    if not losses:
        return pred.sum() * 0.0
    return torch.stack(losses).mean()
