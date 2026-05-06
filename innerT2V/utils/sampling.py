import torch


def time_shift(alpha: float, t: torch.Tensor) -> torch.Tensor:
    return alpha * t / (1 + (alpha - 1) * t)


def get_linear_function(
    x1: float = 464, y1: float = 1, x2: float = 4080, y2: float = 3
) -> callable:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def alphas_cumprod_to_timestep(
    alphas_cumprod: torch.Tensor,
    alphas_cumprod_before_shift: torch.Tensor,
    timesteps_before_shift: torch.Tensor,
) -> torch.Tensor:
    indices = torch.argmin(
        torch.abs(alphas_cumprod_before_shift.view(-1, 1) - alphas_cumprod.view(1, -1)), dim=1)
    return timesteps_before_shift[indices.cpu()]
