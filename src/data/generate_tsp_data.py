"""
python src/data/generate_tsp_data.py --n_cities 20 --n_samples 1000 --split train
"""

from pathlib import Path
import numpy as np

def generate_dataset(num_instances: int, n_nodes: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    return rng.random((num_instances, n_nodes, 2))

def save_dataset(coords: np.ndarray, out_path: str):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, coords=coords)

if __name__ == "__main__":
    base_dir = Path("data/raw/tsp20")

    train = generate_dataset(10000, 20)
    val = generate_dataset(1000, 20)
    test = generate_dataset(1000, 20)

    save_dataset(train, base_dir / "train.npz")
    save_dataset(val, base_dir / "val.npz")
    save_dataset(test, base_dir / "test.npz")