import pytest
import torch


@pytest.mark.requires_gpu
def test_requires_gpu_marker_runs_with_cpu_fallback():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tensor = torch.ones((2, 2), device=device)
    assert float(tensor.sum().cpu()) == 4.0
