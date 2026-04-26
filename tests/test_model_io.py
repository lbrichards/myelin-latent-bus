import pytest
import torch

from latent_bus.model_io import select_device_and_dtype


def test_env_device_and_dtype_override_cpu(monkeypatch):
    monkeypatch.setenv("MYELIN_DEVICE", "cpu")
    monkeypatch.setenv("MYELIN_DTYPE", "float32")

    device, dtype = select_device_and_dtype()

    assert device == "cpu"
    assert dtype == torch.float32


def test_cli_dtype_takes_precedence_over_env(monkeypatch):
    monkeypatch.setenv("MYELIN_DEVICE", "cpu")
    monkeypatch.setenv("MYELIN_DTYPE", "float32")

    device, dtype = select_device_and_dtype(dtype_arg="bf16")

    assert device == "cpu"
    assert dtype == torch.bfloat16


def test_invalid_device_override_raises(monkeypatch):
    monkeypatch.setenv("MYELIN_DEVICE", "not-a-device")

    with pytest.raises(ValueError, match="Unsupported device"):
        select_device_and_dtype()
