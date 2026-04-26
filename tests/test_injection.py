import torch

from latent_bus.injection import answer_start_injection


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        layer = torch.nn.Module()
        layer.mlp = torch.nn.Identity()
        self.layers = torch.nn.ModuleList([layer])


def test_bfloat16_projection_logging_uses_float32_numpy_copy():
    model = DummyModel()
    vector = torch.tensor([1.0, 0.0, 0.0, 0.0])
    handle, effect = answer_start_injection(
        model,
        vector,
        alpha=0.5,
        layer_idx=0,
    )

    try:
        hidden = torch.zeros((1, 2, 4), dtype=torch.bfloat16)
        output = model.layers[0].mlp(hidden)
    finally:
        handle.remove()

    assert output[0, -1, 0].item() == 0.5
    assert effect["proj_before0"] == 0.0
    assert effect["proj_after0"] == 0.5
    assert effect["delta_proj0"] == 0.5


def test_replace_mode_overwrites_last_position_once():
    model = DummyModel()
    vector = torch.tensor([3.0, 4.0, 5.0, 6.0])
    handle, effect = answer_start_injection(
        model,
        vector,
        alpha=0.5,
        layer_idx=0,
        mode="replace",
        burst_steps=1,
    )

    try:
        hidden = torch.ones((1, 2, 4), dtype=torch.float32)
        first = model.layers[0].mlp(hidden)
        second = model.layers[0].mlp(hidden)
    finally:
        handle.remove()

    assert torch.equal(first[0, -1, :], vector)
    assert torch.equal(second[0, -1, :], torch.ones(4))
    assert effect["proj_before0"] == 18.0
    assert effect["proj_after0"] == 86.0
    assert effect["delta_proj0"] == 68.0
