from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


CUDA_PERFORMANCE_NOTE = (
    "CPU模式兼容扫描、净化和ONNX运行；完整净化训练、批量复扫及TensorRT导出"
    "建议使用NVIDIA CUDA，以获得最佳演示性能。"
)


@dataclass(frozen=True)
class ModelSecurityDevicePolicy:
    requested_device: str
    effective_device: str
    cuda_available: bool
    cuda_device_count: int
    cuda_device_index: int | None
    cuda_device_name: str | None
    fallback_reason: str | None
    probe_error: str | None
    cpu_compatible: bool = True
    cuda_recommended: bool = True
    performance_note: str = CUDA_PERFORMANCE_NOTE

    @property
    def uses_cuda(self) -> bool:
        return self.effective_device.startswith("cuda:")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _configured_device(config: Mapping[str, Any] | None) -> Any:
    if not isinstance(config, Mapping):
        return "auto"
    model_security = config.get("model_security")
    if isinstance(model_security, Mapping):
        value = model_security.get("device")
        if value is not None and str(value).strip():
            return value
    inference = config.get("inference")
    if isinstance(inference, Mapping):
        value = inference.get("device")
        if value is not None and str(value).strip():
            return value
    module_a = config.get("module_a")
    if isinstance(module_a, Mapping):
        value = module_a.get("device")
        if value is not None and str(value).strip():
            return value
    return "auto"


def _normalize_requested_device(value: Any) -> tuple[str, int | None, str | None]:
    text = str("auto" if value is None else value).strip().lower()
    if not text or text in {"auto", "automatic", "best"}:
        return "auto", None, None
    if text in {"cpu", "host"}:
        return "cpu", None, None
    if text in {"cuda", "gpu", "nvidia"}:
        return "cuda:0", 0, None
    if text.isdigit():
        return f"cuda:{int(text)}", int(text), None
    if text.startswith("cuda:"):
        raw_index = text.partition(":")[2]
        if raw_index.isdigit():
            return f"cuda:{int(raw_index)}", int(raw_index), None
    return text, None, f"unsupported_requested_device:{text}"


def _probe_cuda(torch_module: Any | None) -> tuple[bool, int, list[str], str | None]:
    try:
        if torch_module is None:
            import torch as torch_module
        available = bool(torch_module.cuda.is_available())
        count = int(torch_module.cuda.device_count()) if available else 0
        names: list[str] = []
        for index in range(count):
            try:
                names.append(str(torch_module.cuda.get_device_name(index)))
            except Exception:
                names.append(f"cuda:{index}")
        return available and count > 0, count, names, None
    except Exception as exc:
        return False, 0, [], f"cuda_probe_failed:{type(exc).__name__}:{exc}"


def resolve_model_security_device(
    config: Mapping[str, Any] | None = None,
    *,
    requested_device: Any | None = None,
    torch_module: Any | None = None,
) -> ModelSecurityDevicePolicy:
    raw_requested = _configured_device(config) if requested_device is None else requested_device
    requested, requested_index, request_error = _normalize_requested_device(raw_requested)
    cuda_available, cuda_count, cuda_names, probe_error = _probe_cuda(torch_module)

    effective = "cpu"
    selected_index: int | None = None
    fallback_reason = request_error
    if request_error is None and requested == "cpu":
        effective = "cpu"
    elif request_error is None and requested == "auto":
        if cuda_available:
            effective = "cuda:0"
            selected_index = 0
        else:
            fallback_reason = probe_error or "cuda_unavailable_auto_cpu_fallback"
    elif request_error is None and requested_index is not None:
        if not cuda_available:
            fallback_reason = probe_error or "requested_cuda_unavailable_cpu_fallback"
        elif requested_index >= cuda_count:
            fallback_reason = (
                f"requested_cuda_device_out_of_range:{requested_index}:available_count={cuda_count}"
            )
        else:
            effective = f"cuda:{requested_index}"
            selected_index = requested_index

    device_name = (
        cuda_names[selected_index]
        if selected_index is not None and selected_index < len(cuda_names)
        else (cuda_names[0] if cuda_names else None)
    )
    return ModelSecurityDevicePolicy(
        requested_device=requested,
        effective_device=effective,
        cuda_available=cuda_available,
        cuda_device_count=cuda_count,
        cuda_device_index=selected_index,
        cuda_device_name=device_name,
        fallback_reason=fallback_reason,
        probe_error=probe_error,
    )
