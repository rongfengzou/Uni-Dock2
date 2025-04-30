from typing import Literal

def run_docking_pipeline(
    json_file_path: str,
    output_dir: str,
    center_x: float,
    center_y: float,
    center_z: float,
    size_x: float = 30.0,
    size_y: float = 30.0,
    size_z: float = 30.0,
    task: Literal['screen', 'score', 'benchmark_one', 'mc'] = 'screen',
    search_mode: Literal['fast', 'balance', 'detail', 'free'] = 'balance',
    exhaustiveness: int = 512,
    randomize: bool = True,
    mc_steps: int = 40,
    opt_steps: int = -1,
    refine_steps: int = 5,
    num_pose: int = 10,
    rmsd_limit: float = 1.0,
    energy_range: float = 5.0,
    seed: int = 1234567,
    use_tor_lib: bool = False,
    constraint_docking: bool = False,
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
        refine_steps: Number of refinement steps
        num_pose: Number of poses to generate
        rmsd_limit: RMSD limit for clustering
        energy_range: Maximum energy difference from best pose
        seed: Random seed
        use_tor_lib: Whether to use torsion library
        constraint_docking: Whether to use constrained docking
        gpu_device_id: GPU device ID to use
    """
    ...
