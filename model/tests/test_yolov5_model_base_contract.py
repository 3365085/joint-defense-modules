from pathlib import Path

import pytest

from defense.module_a.backends.detector_backend import YoloV5DetectorBackend
from defense.runtime.config import (
    apply_custom_model,
    infer_backend_from_model_path,
    infer_model_family_from_model_path,
    normalize_custom_model_options,
    runtime_data_root,
)


def test_backend_inference_from_model_suffixes() -> None:
    assert infer_backend_from_model_path(Path("best.engine")) == "tensorrt"
    assert infer_backend_from_model_path(Path("best.onnx")) == "onnx"
    assert infer_backend_from_model_path(Path("best.pt")) == "pytorch"
    assert infer_backend_from_model_path(Path("best.pth")) == "pytorch"


def test_unknown_custom_model_family_defaults_to_ultralytics(tmp_path: Path) -> None:
    model_path = tmp_path / "best.pt"
    model_path.write_bytes(b"not a torch checkpoint")

    assert infer_model_family_from_model_path(model_path) == "ultralytics"


def test_engine_model_family_uses_path_hints_when_shape_is_unavailable(tmp_path: Path) -> None:
    y5_engine = tmp_path / "baseline_yolov5" / "weights" / "best.engine"
    y8_engine = tmp_path / "baseline_yolov8" / "weights" / "best.engine"

    assert infer_model_family_from_model_path(y5_engine) == "yolov5"
    assert infer_model_family_from_model_path(y8_engine) == "ultralytics"


def test_custom_yolov5_engine_auto_family_is_honored(tmp_path: Path) -> None:
    model_path = tmp_path / "baseline_yolov5" / "weights" / "best.engine"
    config = {"inference": {"backend": "onnx", "model_family": "ultralytics", "artifacts": {}}}
    options = normalize_custom_model_options(
        {"enabled": True, "path": str(model_path), "backend": "auto", "model_family": "auto"}
    )

    resolved = apply_custom_model(config, options)

    assert resolved["backend"] == "tensorrt"
    assert resolved["model_family"] == "yolov5"
    assert resolved["model_family_auto_detected"] is True
    assert config["inference"]["backend"] == "tensorrt"
    assert config["inference"]["model_family"] == "yolov5"
    assert config["inference"]["artifacts"]["engine"] == [str(model_path)]


def test_custom_model_auto_backend_does_not_force_yolov5(tmp_path: Path) -> None:
    model_path = tmp_path / "best.pt"
    model_path.write_bytes(b"not a torch checkpoint")
    config = {"inference": {"backend": "tensorrt", "model_family": "yolov5", "artifacts": {}}}
    options = normalize_custom_model_options(
        {"enabled": True, "path": str(model_path), "backend": "auto", "model_family": "auto"}
    )

    resolved = apply_custom_model(config, options)

    assert resolved["backend"] == "pytorch"
    assert resolved["model_family"] == "ultralytics"
    assert resolved["model_family_auto_detected"] is True
    assert config["inference"]["backend"] == "pytorch"
    assert config["inference"]["model_family"] == "ultralytics"
    assert config["inference"]["artifacts"]["pytorch"] == [str(model_path)]


def test_custom_model_explicit_yolov5_family_is_honored(tmp_path: Path) -> None:
    model_path = tmp_path / "best.pt"
    model_path.write_bytes(b"not a torch checkpoint")
    config = {"inference": {"backend": "tensorrt", "model_family": "yolov5", "artifacts": {}}}
    options = normalize_custom_model_options(
        {"enabled": True, "path": str(model_path), "backend": "pytorch", "model_family": "yolov5"}
    )

    resolved = apply_custom_model(config, options)

    assert resolved["model_family"] == "yolov5"
    assert resolved["model_family_auto_detected"] is False
    assert config["inference"]["backend"] == "pytorch"
    assert config["inference"]["model_family"] == "yolov5"


def test_custom_model_class_names_are_applied_to_runtime_config(tmp_path: Path) -> None:
    model_path = tmp_path / "best.pt"
    model_path.write_bytes(b"not a torch checkpoint")
    config = {"inference": {"backend": "tensorrt", "model_family": "yolov5", "artifacts": {}}}
    options = normalize_custom_model_options(
        {
            "enabled": True,
            "path": str(model_path),
            "backend": "auto",
            "model_family": "yolov5",
            "class_names": ["person", "head", "helmet"],
        }
    )

    resolved = apply_custom_model(config, options)

    assert resolved["class_names"] == ["person", "head", "helmet"]
    assert config["inference"]["class_names"] == ["person", "head", "helmet"]


def test_custom_model_class_names_accepts_operator_text(tmp_path: Path) -> None:
    model_path = tmp_path / "best.pt"
    model_path.write_bytes(b"not a torch checkpoint")
    config = {"inference": {"backend": "tensorrt", "model_family": "yolov5", "artifacts": {}}}
    options = normalize_custom_model_options(
        {
            "enabled": True,
            "path": str(model_path),
            "backend": "auto",
            "model_family": "yolov5",
            "class_names": "person / head / helmet",
        }
    )

    resolved = apply_custom_model(config, options)

    assert resolved["class_names"] == ["person", "head", "helmet"]
    assert config["inference"]["class_names"] == ["person", "head", "helmet"]


def test_custom_model_backend_must_match_file_suffix(tmp_path: Path) -> None:
    model_path = tmp_path / "best.onnx"
    model_path.write_bytes(b"not an onnx file")
    config = {"inference": {"backend": "tensorrt", "model_family": "yolov5", "artifacts": {}}}
    options = normalize_custom_model_options(
        {"enabled": True, "path": str(model_path), "backend": "pytorch", "model_family": "auto"}
    )

    with pytest.raises(ValueError, match="backend does not match"):
        apply_custom_model(config, options)


def test_yolov5_model_base_is_bundled() -> None:
    base = Path(__file__).resolve().parents[1] / "src" / "defense" / "model_bases" / "yolov5_official"
    assert (base / "models" / "experimental.py").is_file()
    assert (base / "utils" / "downloads.py").is_file()
    assert not (base / "data").exists()
    assert not (base / "hubconf.py").exists()
    assert not (base / "export.py").exists()


def test_yolov5_runtime_sets_offline_guards(monkeypatch: pytest.MonkeyPatch) -> None:
    base = Path(__file__).resolve().parents[1] / "src" / "defense" / "model_bases" / "yolov5_official"
    for key in ("YOLOv5_AUTOINSTALL", "YOLOV5_AUTOINSTALL", "YOLOv5_OFFLINE", "YOLOV5_OFFLINE"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("YOLO_CONFIG_DIR", raising=False)

    YoloV5DetectorBackend._configure_yolov5_base_runtime(base)

    runtime_dir = runtime_data_root() / "logs" / "yolov5_runtime"
    assert "0" == __import__("os").environ["YOLOv5_AUTOINSTALL"]
    assert "0" == __import__("os").environ["YOLOV5_AUTOINSTALL"]
    assert "1" == __import__("os").environ["YOLOv5_OFFLINE"]
    assert "1" == __import__("os").environ["YOLOV5_OFFLINE"]
    assert str(runtime_dir) == __import__("os").environ["YOLO_CONFIG_DIR"]


def test_bundled_yolov5_disables_weight_downloads() -> None:
    base = Path(__file__).resolve().parents[1] / "src" / "defense" / "model_bases" / "yolov5_official"
    downloads = (base / "utils" / "downloads.py").read_text(encoding="utf-8")
    general = (base / "utils" / "general.py").read_text(encoding="utf-8")
    common = (base / "models" / "common.py").read_text(encoding="utf-8")
    assert "offline-only" in downloads
    assert "api.github.com" not in downloads + general + common
    assert "releases/download" not in downloads + general + common
    assert "torch.hub.download_url_to_file" not in downloads + general + common
    assert "os.system(" not in downloads + general + common
    assert "requests.get" not in downloads + general + common
    assert "requests.head" not in downloads + general + common
