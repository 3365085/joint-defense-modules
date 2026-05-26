from __future__ import annotations

import argparse
import csv
import json
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from model_security_gate.detox.losses import supervised_yolo_loss, yolov5_decoded_distillation_loss, yolov5_official_loss
from model_security_gate.detox.yolo_dataset import make_yolo_dataloader, move_batch_to_device


@dataclass
class YoloV5OfficialRepairConfig:
    imgsz: int = 640
    batch: int = 4
    device: str | int | None = None
    epochs: int = 2
    lr: float = 2e-5
    weight_decay: float = 5e-4
    max_train_images: int = 0
    max_val_images: int = 256
    grad_clip_norm: float = 5.0
    freeze_backbone_layers: int = 0
    letterbox: bool = False
    seed: int = 0
    teacher_model_path: str | None = None
    distill_weight: float = 0.0
    distill_max_candidates: int = 2048
    loss_mode: str = "proxy"


def _device(value: str | int | None) -> torch.device:
    if value is None:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    text = str(value)
    if text.isdigit():
        return torch.device(f"cuda:{text}")
    return torch.device(text)


def _load_yolov5_checkpoint(path: str | Path, device: torch.device) -> tuple[Any, dict[str, Any]]:
    from defense.runtime.config import _ensure_yolov5_base_importable

    _ensure_yolov5_base_importable()
    ckpt = torch.load(Path(path), map_location=device, weights_only=False)
    if not isinstance(ckpt, dict):
        raise TypeError(f"Expected YOLOv5 checkpoint dict, got {type(ckpt)!r}")
    model = ckpt.get("ema") or ckpt.get("model")
    if model is None:
        raise ValueError("YOLOv5 checkpoint does not contain model or ema")
    model = deepcopy(model).float().to(device)
    model.train()
    for param in model.parameters():
        param.requires_grad_(True)
    return model, ckpt


def _freeze_prefix_layers(model: torch.nn.Module, n_layers: int) -> list[str]:
    frozen: list[str] = []
    if int(n_layers) <= 0:
        return frozen
    modules = getattr(model, "model", None)
    if modules is None:
        return frozen
    for idx, module in enumerate(modules):
        if idx >= int(n_layers):
            break
        for param in module.parameters(recurse=True):
            param.requires_grad_(False)
        frozen.append(str(idx))
    return frozen


def _optimizer(model: torch.nn.Module, cfg: YoloV5OfficialRepairConfig) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters remain after freezing")
    return torch.optim.AdamW(params, lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))


def _freeze_teacher(model: torch.nn.Module) -> torch.nn.Module:
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def _repair_loss(
    model: torch.nn.Module,
    batch: dict[str, Any],
    *,
    teacher: torch.nn.Module | None = None,
    cfg: YoloV5OfficialRepairConfig,
) -> torch.Tensor:
    if str(cfg.loss_mode or "proxy").strip().lower() == "proxy":
        loss = supervised_yolo_loss(model, batch)
    else:
        loss = yolov5_official_loss(model, batch, mode=cfg.loss_mode)
    if teacher is not None and float(cfg.distill_weight) > 0.0:
        was_training = model.training
        model.eval()
        with torch.no_grad():
            teacher_out = teacher(batch["img"])
        student_out = model(batch["img"])
        model.train(was_training)
        distill = yolov5_decoded_distillation_loss(
            student_out,
            teacher_out,
            max_candidates=int(cfg.distill_max_candidates),
        )
        loss = loss + distill * float(cfg.distill_weight)
    return loss


def _epoch_loss(model: torch.nn.Module, loader: Any, device: torch.device, *, teacher: torch.nn.Module | None, cfg: YoloV5OfficialRepairConfig) -> float:
    losses: list[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            loss = _repair_loss(model, batch, teacher=teacher, cfg=cfg)
            losses.append(float(loss.detach().cpu().item()))
    return float(sum(losses) / max(1, len(losses)))


def _save_checkpoint(output_path: Path, source_ckpt: dict[str, Any], model: torch.nn.Module, epoch: int, best_loss: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    saved_model = deepcopy(model).float().cpu()
    saved_model.eval()
    ckpt = dict(source_ckpt)
    ckpt["model"] = saved_model
    ckpt["ema"] = None
    ckpt["optimizer"] = None
    ckpt["epoch"] = int(epoch)
    ckpt["best_fitness"] = -float(best_loss)
    ckpt["updates"] = int(ckpt.get("updates") or 0)
    ckpt["date"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    torch.save(ckpt, output_path)


def run_yolov5_official_repair(
    *,
    model_path: str | Path,
    data_yaml: str | Path,
    output_dir: str | Path,
    cfg: YoloV5OfficialRepairConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or YoloV5OfficialRepairConfig()
    torch.manual_seed(int(cfg.seed))
    device = _device(cfg.device)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, source_ckpt = _load_yolov5_checkpoint(model_path, device)
    teacher = None
    if cfg.teacher_model_path and float(cfg.distill_weight) > 0.0:
        teacher, _ = _load_yolov5_checkpoint(cfg.teacher_model_path, device)
        teacher = _freeze_teacher(teacher)
    frozen_layers = _freeze_prefix_layers(model, cfg.freeze_backbone_layers)
    opt = _optimizer(model, cfg)
    train_loader, info = make_yolo_dataloader(
        data_yaml,
        split="train",
        imgsz=cfg.imgsz,
        batch_size=cfg.batch,
        shuffle=True,
        num_workers=0,
        max_images=cfg.max_train_images if cfg.max_train_images and cfg.max_train_images > 0 else None,
        letterbox=cfg.letterbox,
    )
    val_loader = None
    if int(cfg.max_val_images) != 0:
        val_loader, _ = make_yolo_dataloader(
            data_yaml,
            split="val",
            imgsz=cfg.imgsz,
            batch_size=cfg.batch,
            shuffle=False,
            num_workers=0,
            max_images=cfg.max_val_images if cfg.max_val_images and cfg.max_val_images > 0 else None,
            letterbox=cfg.letterbox,
        )

    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    started = time.perf_counter()
    for epoch in range(1, int(cfg.epochs) + 1):
        model.train()
        train_losses: list[float] = []
        for batch in train_loader:
            batch = move_batch_to_device(batch, device)
            opt.zero_grad(set_to_none=True)
            loss = _repair_loss(model, batch, teacher=teacher, cfg=cfg)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            if float(cfg.grad_clip_norm) > 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], float(cfg.grad_clip_norm))
            opt.step()
            train_losses.append(float(loss.detach().cpu().item()))
        train_loss = float(sum(train_losses) / max(1, len(train_losses)))
        val_loss = _epoch_loss(model, val_loader, device, teacher=teacher, cfg=cfg) if val_loader is not None else train_loss
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "steps": len(train_losses)}
        history.append(row)
        _save_checkpoint(last_path, source_ckpt, model, epoch, val_loss)
        if val_loss <= best_loss:
            best_loss = val_loss
            _save_checkpoint(best_path, source_ckpt, model, epoch, val_loss)

    history_csv = output_dir / "repair_history.csv"
    with history_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "steps"])
        writer.writeheader()
        writer.writerows(history)
    manifest = {
        "algorithm": "yolov5_official_proxy_repair",
        "model_path": str(model_path),
        "data_yaml": str(data_yaml),
        "output_dir": str(output_dir),
        "best_model_path": str(best_path),
        "last_model_path": str(last_path),
        "history_csv_path": str(history_csv),
        "elapsed_s": time.perf_counter() - started,
        "config": asdict(cfg),
        "data": {
            "root": str(info.root),
            "train_images": len(info.train_images),
            "val_images": len(info.val_images),
            "names": info.names,
        },
        "frozen_layers": frozen_layers,
        "history": history,
    }
    manifest_path = output_dir / "repair_manifest.json"
    manifest["manifest_path"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair a bundled YOLOv5-official checkpoint without architecture migration.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-yaml", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--max-train-images", type=int, default=0)
    parser.add_argument("--max-val-images", type=int, default=256)
    parser.add_argument("--grad-clip-norm", type=float, default=5.0)
    parser.add_argument("--freeze-backbone-layers", type=int, default=0)
    parser.add_argument("--letterbox", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--teacher-model", default=None)
    parser.add_argument("--distill-weight", type=float, default=0.0)
    parser.add_argument("--distill-max-candidates", type=int, default=2048)
    parser.add_argument("--loss-mode", default="proxy", choices=["proxy", "compute", "official", "yolov5", "combined", "hybrid"])
    args = parser.parse_args()
    payload = run_yolov5_official_repair(
        model_path=args.model,
        data_yaml=args.data_yaml,
        output_dir=args.output_dir,
        cfg=YoloV5OfficialRepairConfig(
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            max_train_images=args.max_train_images,
            max_val_images=args.max_val_images,
            grad_clip_norm=args.grad_clip_norm,
            freeze_backbone_layers=args.freeze_backbone_layers,
            letterbox=args.letterbox,
            seed=args.seed,
            teacher_model_path=args.teacher_model,
            distill_weight=args.distill_weight,
            distill_max_candidates=args.distill_max_candidates,
            loss_mode=args.loss_mode,
        ),
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
