import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

try:
    from torch_geometric.data import HeteroData
except Exception as exc:  # pragma: no cover - optional dependency
    HeteroData = None

from utils.arguments import load_params
from utils.common import triangles_to_edges, add_field_to_pyg_batch
from utils.defaults import DEFAULTS
from utils.validation import load_runner_from_checkpoint


@dataclass
class DefaultsOverride:
    data_root: Optional[str] = None
    aux_data: Optional[str] = None
    project_dir: Optional[str] = None
    config_dir: Optional[str] = None
    cmu_root: Optional[str] = None
    results_dir: Optional[str] = None


def override_defaults(override: DefaultsOverride) -> None:
    if override.data_root:
        DEFAULTS.data_root = override.data_root
        if not override.aux_data:
            DEFAULTS.aux_data = str(Path(override.data_root) / "aux_data")
    if override.aux_data:
        DEFAULTS.aux_data = override.aux_data
    if override.project_dir:
        DEFAULTS.project_dir = override.project_dir
    if override.config_dir:
        DEFAULTS.config_dir = override.config_dir
    if override.cmu_root:
        DEFAULTS.CMU_root = override.cmu_root
    if override.results_dir:
        DEFAULTS.results_dir = override.results_dir


class HoodMinimal:
    def __init__(
        self,
        config_name: str = "hood_final",
        checkpoint_path: Optional[str] = None,
        device: str = "cpu",
        defaults: Optional[DefaultsOverride] = None,
    ) -> None:
        if defaults:
            override_defaults(defaults)

        modules, experiment_config = load_params(config_name)

        if checkpoint_path is None:
            checkpoint_path = str(Path(DEFAULTS.data_root) / "trained_models" / f"{config_name}.pth")

        runner_module, runner = load_runner_from_checkpoint(checkpoint_path, modules, experiment_config)

        self.device = device
        self.runner_module = runner_module
        self.runner = runner.to(device)
        self.model = runner.model

    def forward(self, sample):
        sample = sample.to(self.device)
        return self.model(sample)

    def rollout(self, sequence, n_steps=10):
        sequence = sequence.to(self.device)
        return self.runner.valid_rollout(sequence, bare=True, n_steps=n_steps)


def build_dummy_sample(
    n_cloth: int = 4,
    n_obstacle: int = 4,
    n_coarse_levels: int = 1,
    device: str = "cpu",
):
    if HeteroData is None:
        raise RuntimeError("torch_geometric is required to build a HeteroData sample.")

    sample = HeteroData()

    # Cloth nodes
    cloth_pos = torch.randn(n_cloth, 3, device=device)
    cloth_prev_pos = cloth_pos - 0.01
    cloth_target_pos = cloth_pos + 0.01
    cloth_rest_pos = cloth_pos.clone()

    sample["cloth"].pos = cloth_pos
    sample["cloth"].prev_pos = cloth_prev_pos
    sample["cloth"].target_pos = cloth_target_pos
    sample["cloth"].rest_pos = cloth_rest_pos

    # Obstacle nodes
    obstacle_pos = torch.randn(n_obstacle, 3, device=device)
    obstacle_prev_pos = obstacle_pos - 0.01
    obstacle_target_pos = obstacle_pos + 0.01

    sample["obstacle"].pos = obstacle_pos
    sample["obstacle"].prev_pos = obstacle_prev_pos
    sample["obstacle"].target_pos = obstacle_target_pos

    # Faces (dummy triangles)
    cloth_faces = torch.tensor([[0, 1, 2], [1, 2, 3]], device=device, dtype=torch.long)
    obstacle_faces = torch.tensor([[0, 1, 2], [1, 2, 3]], device=device, dtype=torch.long)
    sample["cloth"].faces_batch = cloth_faces.T
    sample["obstacle"].faces_batch = obstacle_faces.T

    # Mesh edges from faces
    mesh_edges = triangles_to_edges(cloth_faces.unsqueeze(0))
    sample["cloth", "mesh_edge", "cloth"].edge_index = mesh_edges

    # Coarse edges (simple chain per level)
    for i in range(n_coarse_levels):
        if n_cloth >= 2:
            edges = torch.stack(
                [torch.arange(0, n_cloth - 1), torch.arange(1, n_cloth)],
                dim=0,
            ).to(device)
        else:
            edges = torch.zeros((2, 0), dtype=torch.long, device=device)
        sample["cloth", f"coarse_edge{i}", "cloth"].edge_index = edges

    # Vertex types and levels
    sample["cloth"].vertex_type = torch.zeros((n_cloth, 1), dtype=torch.long, device=device)
    sample["obstacle"].vertex_type = torch.ones((n_obstacle, 1), dtype=torch.long, device=device)
    sample["cloth"].vertex_level = torch.zeros((n_cloth, 1), dtype=torch.long, device=device)
    sample["obstacle"].vertex_level = torch.zeros((n_obstacle, 1), dtype=torch.long, device=device)

    # Material params (normalized inputs expected by the model)
    add_field_to_pyg_batch(sample, "lame_mu_input", torch.tensor([[0.0]], device=device), "cloth", reference_key=None, one_per_sample=True)
    add_field_to_pyg_batch(sample, "lame_lambda_input", torch.tensor([[0.0]], device=device), "cloth", reference_key=None, one_per_sample=True)
    add_field_to_pyg_batch(sample, "bending_coeff_input", torch.tensor([[0.0]], device=device), "cloth", reference_key=None, one_per_sample=True)

    # Timestep and velocity
    add_field_to_pyg_batch(sample, "timestep", torch.tensor([[1.0]], device=device), "cloth", reference_key=None, one_per_sample=True)

    cloth_velocity = cloth_pos - cloth_prev_pos
    obstacle_velocity = obstacle_pos - obstacle_prev_pos
    obstacle_next_velocity = obstacle_target_pos - obstacle_pos

    add_field_to_pyg_batch(sample, "velocity", cloth_velocity, "cloth", "pos")
    add_field_to_pyg_batch(sample, "velocity", obstacle_velocity, "obstacle", "pos")
    add_field_to_pyg_batch(sample, "next_velocity", obstacle_next_velocity, "obstacle", "pos")

    return sample


if __name__ == "__main__":
    device = os.environ.get("HOOD_DEVICE", "cpu")
    config_name = os.environ.get("HOOD_CONFIG", "hood_final")
    checkpoint_path = os.environ.get("HOOD_CHECKPOINT")

    defaults = DefaultsOverride(
        data_root=os.environ.get("HOOD_DATA_ROOT"),
        aux_data=os.environ.get("HOOD_AUX_DATA"),
        project_dir=os.environ.get("HOOD_PROJECT_DIR"),
        config_dir=os.environ.get("HOOD_CONFIG_DIR"),
        cmu_root=os.environ.get("HOOD_CMU_ROOT"),
        results_dir=os.environ.get("HOOD_RESULTS_DIR"),
    )

    model = HoodMinimal(
        config_name=config_name,
        checkpoint_path=checkpoint_path,
        device=device,
        defaults=defaults,
    )

    sample = build_dummy_sample(device=device)
    with torch.no_grad():
        out = model.forward(sample)
    print("OK: forward ran, pred_pos shape:", out["cloth"].pred_pos.shape)
