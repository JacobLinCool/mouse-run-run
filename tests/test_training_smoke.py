import math

from mouse_run_run.env import GridWorldConfig
from mouse_run_run.train import TrainConfig, train


def test_tiny_training_run_writes_checkpoint_and_finite_metrics(tmp_path) -> None:
    checkpoint = tmp_path / "latest.safetensors"

    metrics = train(
        TrainConfig(
            updates=1,
            batch_size=2,
            hidden_size=8,
            device="cpu",
            log_every=1,
            checkpoint=checkpoint,
            env=GridWorldConfig(max_steps=5),
        )
    )

    assert checkpoint.exists()
    assert math.isfinite(metrics.chaser_return)
    assert math.isfinite(metrics.explorer_return)
    assert math.isfinite(metrics.value_loss)
