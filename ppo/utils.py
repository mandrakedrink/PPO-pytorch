import numpy as np
import torch

def get_gae(
    rewards: list,
    values: list,
    is_terminals: list,
    gamma: float,
    lamda: float,
    ):
    """
    Takes: lists of rewards, state values, and 1-dones.
    Returns: list with generalized adversarial estimators.
    More details - https://arxiv.org/pdf/1506.02438.pdf.
    """
    gae = 0
    returns = []
    for i in reversed(range(len(rewards))):
        delta = (rewards[i] + gamma * values[i + 1] * is_terminals[i] - values[i])
        gae = delta + gamma * lamda * is_terminals[i] * gae
        returns.insert(0, gae + values[i])

    return returns

def trajectories_data_generator(
    states: torch.Tensor,
    actions: torch.Tensor,
    returns: torch.Tensor,
    log_probs: torch.Tensor,
    values: torch.Tensor,
    advantages: torch.Tensor,
    batch_size,
    num_epochs,
    ):
    """data-generator."""
    data_len = states.size(0)
    for _ in range(num_epochs):
        for _ in range(data_len // batch_size):
            ids = np.random.choice(data_len, batch_size)
            yield states[ids, :], actions[ids], returns[ids], log_probs[ids], values[ids], advantages[ids]

