"""
Global configuration: random seeds, device selection, and experiment hyperparameters.
"""
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict, Any
from pathlib import Path
import random
import numpy as np
import torch
import yaml
import warnings


SEED = 2049
"""Global random seed for reproducibility across all libraries."""

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if DEVICE == "cpu":
    warnings.warn(
            "\n" + "=" * 72 + "\n"
            "AVISO: PyTorch nao detectou GPU. Os experimentos rodarao em CPU\n")


def seed_worker(worker_id: int):
    """
    Worker initialization function for DataLoader reproducibility.

    Args:
        worker_id: index of the DataLoader worker process
    """
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def log_step(msg: str):
    """Prints a message with flush=True for unbuffered logging."""
    print(msg, flush=True)


@dataclass
class ExperimentConfig:
    """
    Holds all hyperparameters for a single FEDMARL experiment run.

    Loaded from `configs/experiments.yaml` via `ExperimentConfig.load(...)`.
    Every field maps 1:1 to a key in the YAML config.
    """
    # ---- Identification ----
    exp_name: str = "main"

    # ---- Federated setup ----
    rounds: int = 500
    n_clients: int = 50
    k_select: int = 15
    dir_alpha: float = 0.3

    # ---- Initial attack ----
    initial_flip_fraction: float = 0.4
    flip_add_fraction: float = 0.0
    attack_rounds: List[int] = field(default_factory=lambda: [600])
    flip_rate_initial: float = 1.0
    flip_rate_new_attack: float = 0.0

    # ---- Attack type ----
    targeted_only_map_classes: bool = True
    target_map: Optional[Dict[int, int]] = None

    # ---- Local training ----
    max_per_client: int = 2500
    local_lr: float = 0.01
    local_steps: int = 10
    probe_batches: int = 10

    # ---- Server gradient EMA ----
    mom_beta: float = 0.90

    # ---- Reward ----
    reward_window_W: int = 5

    # ---- MARL / VDN ----
    marl_eps: float = 0.15
    marl_swap_m: int = 2
    marl_lr: float = 1e-3
    marl_gamma: float = 0.90
    marl_hidden: int = 128
    marl_target_sync_every: int = 20
    warmup_transitions: int = 50
    start_train_round: int = 50
    updates_per_round: int = 50
    train_every: int = 1

    # ---- Replay buffer ----
    buf_size: int = 20000
    batch_base: int = 32
    batch_max: int = 256
    batch_buffer_ratio: int = 4

    # ---- PER ----
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0
    per_beta_steps: int = 4000
    per_eps: float = 1e-3

    # ---- Evaluation ----
    val_shuffle: bool = False
    val_per_class: int = 200
    eval_max_batches: int = 20
    print_every: int = 1
    print_advfo_every: int = 20

    # ---- Output ----
    out_dir: str = "."
    save_results: bool = True

    # ---------- Serialization ----------

    def dump(self, path: str | Path) -> None:
        """Write the resolved config to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(asdict(self), f, sort_keys=False)

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        valid = set(cls.__dataclass_fields__)
        unknown = set(data) - valid
        if unknown:
            raise ValueError(
                f"Unknown config keys: {sorted(unknown)}. "
                f"Valid keys: {sorted(valid)}"
            )
        return cls(**data)

    @classmethod
    def load(
        cls,
        config_path: str | Path,
        experiment: Optional[str] = None,
        cli_overrides: Optional[Dict[str, Any]] = None,
    ) -> "ExperimentConfig":
        """
        Load a config from a unified YAML file.

        The YAML must have the structure:
            defaults: { ...all hyperparameters... }
            experiments:
                exp1: { ...keys that differ from defaults... }
                exp2: { ... }
                ...

        Resolution order: defaults -> experiments[experiment] -> cli_overrides.
        Pass `experiment=None` to use defaults only.
        """
        with open(config_path) as f:
            doc = yaml.safe_load(f) or {}

        if "defaults" not in doc:
            raise ValueError(
                f"Config {config_path} must contain a top-level 'defaults' block."
            )
        data = dict(doc["defaults"])

        if experiment is not None:
            experiments = doc.get("experiments", {}) or {}
            if experiment not in experiments:
                available = sorted(experiments.keys())
                raise ValueError(
                    f"Experiment {experiment!r} not found in {config_path}. "
                    f"Available: {available}"
                )
            preset = experiments[experiment] or {}
            if "exp_name" not in preset:
                data["exp_name"] = experiment
            data.update(preset)

        if cli_overrides:
            data.update(cli_overrides)

        return cls._from_dict(data)

    @classmethod
    def list_experiments(cls, config_path: str | Path) -> List[str]:
        """Return the experiment names declared in the unified YAML."""
        with open(config_path) as f:
            doc = yaml.safe_load(f) or {}
        return sorted((doc.get("experiments") or {}).keys())
