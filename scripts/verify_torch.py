import platform

import torch


def main() -> None:
    hidden_size = 256
    input_size = 200
    action_size = 4
    batch_size = 1

    x_t = torch.randn(batch_size, input_size)
    h_prev = torch.randn(batch_size, hidden_size)

    w_input = torch.randn(hidden_size, input_size)
    b_input = torch.randn(hidden_size)
    w_rec = torch.randn(hidden_size, hidden_size)
    b_rec = torch.randn(hidden_size)
    w_action = torch.randn(action_size, hidden_size)

    h_t = torch.relu(x_t @ w_input.T + b_input + h_prev @ w_rec.T + b_rec)
    neural_action_subspace = h_t @ w_action.T @ w_action
    action_logits = h_t @ w_action.T
    action_probs = torch.softmax(action_logits, dim=-1)

    print(f"python {platform.python_version()}")
    print(f"torch {torch.__version__}")
    print(f"mps_available {torch.backends.mps.is_available()}")
    print(f"h_t {tuple(h_t.shape)}")
    print(f"neural_action_subspace {tuple(neural_action_subspace.shape)}")
    print(f"action_probs {action_probs.detach().round(decimals=4).tolist()}")


if __name__ == "__main__":
    main()
