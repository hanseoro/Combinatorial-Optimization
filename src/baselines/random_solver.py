"""
Random baseline solver for Euclidean Traveling Salesman Problem (TSP).

This module provides a simple baseline that generates a random valid tour
(permutation of node indices) for each TSP instance.

Conventions
-----------
- A TSP instance is represented by coordinates of shape (n_nodes, 2).
- A batch of instances is represented by shape (batch_size, n_nodes, 2).
- Tours are represented as permutations of node indices with shape (n_nodes,)
  for a single instance or (batch_size, n_nodes) for a batch.
- Tours are implicitly closed when computing tour length.

"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.evaluation.metrics import compute_batch_tour_lengths, compute_tour_length


@dataclass(frozen=True)
class RandomSolverResult:
    """
    Output container for a random TSP solve operation.

    Attributes:
        tours:
            Array of shape (batch_size, n_nodes) containing random tours.
        costs:
            Array of shape (batch_size,) containing the corresponding tour costs.
    """

    tours: np.ndarray
    costs: np.ndarray


def _ensure_numpy_array(array: Any, *, name: str) -> np.ndarray:
    """
    Convert input to a NumPy array.

    Args:
        array:
            Input object to convert.
        name:
            Argument name for error messages.

    Returns:
        NumPy array.
    """
    try:
        return np.asarray(array)
    except Exception as exc:  # pragma: no cover
        raise TypeError(f"Could not convert '{name}' to a NumPy array.") from exc


def _validate_single_coords(coords: np.ndarray) -> None:
    """
    Validate a single TSP instance coordinate array.

    Args:
        coords:
            Array of shape (n_nodes, 2).

    Raises:
        ValueError:
            If the coordinates are malformed.
    """
    if coords.ndim != 2:
        raise ValueError(
            f"'coords' must have shape (n_nodes, 2), but got ndim={coords.ndim}."
        )
    if coords.shape[1] != 2:
        raise ValueError(
            f"'coords' must have shape (n_nodes, 2), but got shape {coords.shape}."
        )
    if coords.shape[0] <= 0:
        raise ValueError("'coords' must contain at least one node.")
    if not np.all(np.isfinite(coords)):
        raise ValueError("'coords' contains non-finite values.")


def _validate_batch_coords(coords_batch: np.ndarray) -> None:
    """
    Validate a batch of TSP instance coordinate arrays.

    Args:
        coords_batch:
            Array of shape (batch_size, n_nodes, 2).

    Raises:
        ValueError:
            If the coordinates are malformed.
    """
    if coords_batch.ndim != 3:
        raise ValueError(
            "'coords_batch' must have shape (batch_size, n_nodes, 2), "
            f"but got ndim={coords_batch.ndim}."
        )
    if coords_batch.shape[2] != 2:
        raise ValueError(
            "'coords_batch' must have shape (batch_size, n_nodes, 2), "
            f"but got shape {coords_batch.shape}."
        )
    if coords_batch.shape[0] <= 0:
        raise ValueError("'coords_batch' must contain at least one instance.")
    if coords_batch.shape[1] <= 0:
        raise ValueError("'coords_batch' must contain at least one node per instance.")
    if not np.all(np.isfinite(coords_batch)):
        raise ValueError("'coords_batch' contains non-finite values.")


class RandomTSPSolver:
    """
    Random baseline solver for Euclidean TSP.

    This solver generates a random permutation of nodes for each instance.
    It can be used for quick benchmarking and debugging.

    Args:
        seed:
            Optional random seed for reproducibility.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)

    def solve(self, coords: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Generate a random tour for a single TSP instance.

        Args:
            coords:
                Array of shape (n_nodes, 2).

        Returns:
            A tuple (tour, cost), where:
            - tour has shape (n_nodes,)
            - cost is the total Euclidean tour length
        """
        coords = _ensure_numpy_array(coords, name="coords").astype(np.float64, copy=False)
        _validate_single_coords(coords)

        n_nodes = coords.shape[0]
        tour = self._rng.permutation(n_nodes).astype(np.int64, copy=False)
        cost = compute_tour_length(coords, tour, validate=False)

        return tour, cost

    def solve_batch(self, coords_batch: np.ndarray) -> RandomSolverResult:
        """
        Generate random tours for a batch of TSP instances.

        Args:
            coords_batch:
                Array of shape (batch_size, n_nodes, 2).

        Returns:
            RandomSolverResult with:
            - tours of shape (batch_size, n_nodes)
            - costs of shape (batch_size,)
        """
        coords_batch = _ensure_numpy_array(coords_batch, name="coords_batch").astype(
            np.float64, copy=False
        )
        _validate_batch_coords(coords_batch)

        batch_size, n_nodes, _ = coords_batch.shape

        tours = np.empty((batch_size, n_nodes), dtype=np.int64)
        for i in range(batch_size):
            tours[i] = self._rng.permutation(n_nodes)

        costs = compute_batch_tour_lengths(
            coords_batch=coords_batch,
            tours_batch=tours,
            validate=False,
        )

        return RandomSolverResult(tours=tours, costs=costs)

    def __call__(self, coords: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Alias for solve() so the solver can be used like a callable.

        Args:
            coords:
                Array of shape (n_nodes, 2).

        Returns:
            A tuple (tour, cost).
        """
        return self.solve(coords)


def solve_random_tsp(coords: np.ndarray, seed: int | None = None) -> tuple[np.ndarray, float]:
    """
    Convenience function for solving a single TSP instance with a random tour.

    Args:
        coords:
            Array of shape (n_nodes, 2).
        seed:
            Optional random seed for reproducibility.

    Returns:
        A tuple (tour, cost).
    """
    solver = RandomTSPSolver(seed=seed)
    return solver.solve(coords)


def solve_random_tsp_batch(coords_batch: np.ndarray, seed: int | None = None) -> RandomSolverResult:
    """
    Convenience function for solving a batch of TSP instances with random tours.

    Args:
        coords_batch:
            Array of shape (batch_size, n_nodes, 2).
        seed:
            Optional random seed for reproducibility.

    Returns:
        RandomSolverResult containing tours and costs.
    """
    solver = RandomTSPSolver(seed=seed)
    return solver.solve_batch(coords_batch)


__all__ = [
    "RandomSolverResult",
    "RandomTSPSolver",
    "solve_random_tsp",
    "solve_random_tsp_batch",
]