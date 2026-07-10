import pytest
import torch

from mouse_run_run.policy import build_policy


@pytest.mark.parametrize("architecture", ["rnn", "mlp", "ssm", "transformer"])
def test_policy_forward_and_sequence_shapes(architecture: str) -> None:
    torch.manual_seed(0)
    batch_size = 2
    steps = 3
    input_size = 18
    hidden_size = 8
    policy = build_policy(architecture, input_size, hidden_size=hidden_size)
    observation = torch.zeros(batch_size, input_size)
    observation[:, 0] = 1.0

    state = policy.initial_hidden(batch_size, torch.device("cpu"))
    output = policy(observation, state)

    assert output.logits.shape == (batch_size, 4)
    assert output.value.shape == (batch_size,)
    assert output.hidden.shape == (batch_size, hidden_size)

    sequence = observation.repeat(steps, 1, 1)
    logits, values, final_hidden = policy.sequence(sequence)

    assert logits.shape == (steps, batch_size, 4)
    assert values.shape == (steps, batch_size)
    assert final_hidden.shape == (batch_size, hidden_size)


def test_unknown_policy_architecture_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown architecture"):
        build_policy("not-a-policy", input_size=18, hidden_size=8)


def test_official_rnn_uses_pytorch_default_initialization() -> None:
    torch.manual_seed(0)
    policy = build_policy(
        "rnn",
        input_size=18,
        hidden_size=8,
        rnn_initialization="pytorch_default",
    )

    # PyTorch's native RNN/Linear initialization includes non-zero biases;
    # the modern runner's custom initialization zeros them.
    assert torch.count_nonzero(policy.rnn.bias_ih_l0).item() > 0
    assert torch.count_nonzero(policy.action_layer.bias).item() > 0
