from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DatasetConfig:
    theta_low: float = -3.0
    theta_high: float = 3.0
    sigma_x1: float = 0.30
    sigma_x2: float = 0.22
    rho: float = 0.15
    seed: int = 7


@dataclass
class ScoreRunConfig:
    epochs: int = 120
    batch_size: int = 256
    lr: float = 1e-3
    hidden_dim: int = 128
    depth: int = 3
    sigma_alpha_list: list[float] = field(default_factory=lambda: [0.08, 0.06, 0.045, 0.03, 0.02])
    n_train: int = 28000
    n_eval: int = 18000
    fd_delta: float = 0.03
    n_bins: int = 35
    min_bin_count: int = 80
    eval_margin: float = 0.30
    log_every: int = 25
    output_dir: str = "outputs_step3_multi_sigma"
    device: str = "cpu"


@dataclass
class DecoderRunConfig:
    epsilon: float = 0.12
    fd_delta: float = 0.03
    n_bins: int = 35
    eval_margin: float = 0.30
    n_train_local: int = 1200
    n_eval_local: int = 1200
    epochs: int = 80
    batch_size: int = 256
    lr: float = 1e-3
    hidden_dim: int = 64
    depth: int = 2
    log_every: int = 5
    output_dir: str = "outputs_step4_decoder"
    device: str = "cpu"
