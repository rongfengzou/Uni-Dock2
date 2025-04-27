from typing import Literal


def run_docking_pipeline(
    json_file_path: str,
    output_dir: str,
    center_x: float,
    center_y: float,
    center_z: float,
    size_x: float,
    size_y: float,
    size_z: float,
    task: Literal["screen", "score", "benchmark_one", "mc"] = "screen",
    search_mode: Literal["fast", "balance", "detail", "free"] = "balance",
    exhaustiveness: int = 128,
    randomize: bool = True,
    mc_steps: int = 20,
    opt_steps: int = 10,
    tor_prec: float = 0.3,
    box_prec: float = 1.0,
    refine_steps: int = 5,
    num_pose: int = 1,
    rmsd_limit: float = 1.0,
    energy_range: float = 3.0,
    seed: int = 12345,
    constraint_docking: bool = False,
    use_tor_lib: bool = True,
    gpu_device_id: int = 0
) -> None:
    """
    Run docking pipeline using the Uni-Dock2 engine.
    
    Args:
        json_file_path: Path to the input JSON file
        output_dir: Directory to store output files
        center_x: X-coordinate of box center
        center_y: Y-coordinate of box center
        center_z: Z-coordinate of box center
        size_x: Box size in X dimension
        size_y: Box size in Y dimension
        size_z: Box size in Z dimension
        task: Task type ("screen", "score", "benchmark_one", or "mc")
        search_mode: Search mode ("fast", "balance", "detail", or "free")
        exhaustiveness: Number of independent runs
        randomize: Whether to randomize starting positions
        mc_steps: Number of Monte Carlo steps
        opt_steps: Number of optimization steps
        tor_prec: Torsion angle precision
        box_prec: Box precision
        refine_steps: Number of refinement steps
        num_pose: Number of poses to generate
        rmsd_limit: RMSD limit for clustering
        energy_range: Maximum energy difference from best pose
        seed: Random seed
        constraint_docking: Whether to use constrained docking
        use_tor_lib: Whether to use torsion library
        gpu_device_id: GPU device ID to use
    """
    ...
