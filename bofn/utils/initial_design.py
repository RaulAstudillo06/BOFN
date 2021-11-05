import torch


def generate_initial_design(num_samples: int, input_dim: int, seed=None):
    # generate training data
    if seed is not None:
        old_state = torch.random.get_rng_state()
        torch.manual_seed(seed)
        X = torch.rand([num_samples, input_dim])
        torch.random.set_rng_state(old_state)
    else:
        X = torch.rand([num_samples, input_dim])
    return X
