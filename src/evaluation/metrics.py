"""
Evaluation metrics for Traveling Salesman Problem (TSP) experiments.

This module provides pure, reusable metric functions for:
- validating predicted tours
- computing tour lengths
- computing optimality gaps against reference solutions
- aggregating batch-level summaries for reporting

Conventions
-----------
- Coordinates are expected to have shape (n_nodes, 2) for a single instance
  or (batch_size, n_nodes, 2) for a batch.
- Tours are represented as permutations of node indices with shape (n_nodes,)
  for a single instance or (batch_size, n_nodes) for a batch.
- Tours are closed
"""

from dataclasses import dataclass
from typing import Any

import numpy as np


EPS = 1e-12


@dataclass(frozen=True)
class TourValidationResult:
    """
    Result of validating a TSP tour.

    Attributes:
        is_valid:
            True if the tour is a valid permutation of {0, ..., n_nodes - 1}.
        reason:
            Human-readable reason for invalidity, or None if the tour is valid.
    """

    is_valid: bool
    reason: str | None = None


def _ensure_numpy_array(array: Any, *, name: str) -> np.ndarray:
    """
    Convert an input to a NumPy array.

    Args:
        array:
            Input object to convert.
        name:
            Name of the argument for error messages.

    Returns:
        NumPy array view/copy of the input.
    """
    try:
        return np.asarray(array)
    except Exception as exc:
        raise TypeError(f"Could not convert '{name}' to a NumPy array.") from exc


def _validate_single_coords(coords: np.ndarray) -> None:
    """
    Validate the shape and contents of a single coordinate array.

    Args:
        coords:
            Array of shape (n_nodes, 2).

    Raises:
        ValueError:
            If coords does not have the expected shape or contains invalid values.
    """
    if coords.ndim != 2:
        raise ValueError(
            f"'coords' must have shape (n_nodes, 2), but got ndim={coords.ndim}."
        )
    if coords.shape[1] != 2:
        raise ValueError(
            f"'coords' must have shape (n_nodes, 2), but got shape {coords.shape}."
        )
    if coords.shape[0] == 0:
        raise ValueError("'coords' must contain at least one node.")
    if not np.all(np.isfinite(coords)):
        raise ValueError("'coords' contains non-finite values.")


def _validate_batch_coords(coords_batch: np.ndarray) -> None:
    """
    Validate the shape and contents of a batch coordinate array.

    Args:
        coords_batch:
            Array of shape (batch_size, n_nodes, 2).

    Raises:
        ValueError:
            If coords_batch does not have the expected shape or contains invalid values.
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
    if coords_batch.shape[0] == 0:
        raise ValueError("'coords_batch' must contain at least one instance.")
    if coords_batch.shape[1] == 0:
        raise ValueError("'coords_batch' must contain at least one node per instance.")
    if not np.all(np.isfinite(coords_batch)):
        raise ValueError("'coords_batch' contains non-finite values.")


def validate_tour(tour: np.ndarray, n_nodes: int) -> TourValidationResult:
    """
    Validate that a tour is a proper permutation of node indices.

    A valid TSP tour must:
    - have length n_nodes
    - contain integer-valued indices
    - contain only values in [0, n_nodes - 1]
    - visit every node exactly once

    Args:
        tour:
            Array-like of shape (n_nodes,).
        n_nodes:
            Number of nodes expected in the instance.

    Returns:
        TourValidationResult describing validity and, if invalid, the reason.
    """
    tour = _ensure_numpy_array(tour, name="tour")

    if not isinstance(n_nodes, int) or n_nodes <= 0:
        return TourValidationResult(
            is_valid=False,
            reason="'n_nodes' must be a positive integer.",
        )

    if tour.ndim != 1:
        return TourValidationResult(
            is_valid=False,
            reason=f"'tour' must be 1D, but got ndim={tour.ndim}.",
        )

    if len(tour) != n_nodes:
        return TourValidationResult(
            is_valid=False,
            reason=f"'tour' length {len(tour)} does not match n_nodes={n_nodes}.",
        )

    if not np.all(np.isfinite(tour)):
        return TourValidationResult(
            is_valid=False,
            reason="'tour' contains non-finite values.",
        )

    if not np.all(np.equal(tour, np.floor(tour))):
        return TourValidationResult(
            is_valid=False,
            reason="'tour' must contain integer-valued node indices.",
        )

    tour_int = tour.astype(np.int64, copy=False)

    if np.any(tour_int < 0) or np.any(tour_int >= n_nodes):
        return TourValidationResult(
            is_valid=False,
            reason=f"'tour' contains indices outside the valid range [0, {n_nodes - 1}].",
        )

    unique_count = np.unique(tour_int).size
    if unique_count != n_nodes:
        return TourValidationResult(
            is_valid=False,
            reason="The tour must visit each node exactly once.",
        )

    return TourValidationResult(is_valid=True, reason=None)


def is_valid_tour(tour: np.ndarray, n_nodes: int) -> bool:
    """
    Return True if the tour is a valid TSP permutation, else False.

    Args:
        tour:
            Array-like of shape (n_nodes,).
        n_nodes:
            Number of nodes expected in the instance.

    Returns:
        Boolean validity indicator.
    """
    return validate_tour(tour, n_nodes).is_valid


def compute_tour_length(coords: np.ndarray, tour: np.ndarray, *, validate: bool = True,) -> float:
    """
    Compute the Euclidean length of a single closed TSP tour.

    Args:
        coords:
            Array of shape (n_nodes, 2).
        tour:
            Array of shape (n_nodes,), representing a permutation of node indices.
        validate:
            If True, validate the tour before computing the length.

    Returns:
        Total closed-tour length.

    Raises:
        ValueError:
            If input shapes are invalid or the tour is invalid when validate=True.
    """
    coords = _ensure_numpy_array(coords, name="coords").astype(np.float64, copy=False)
    tour = _ensure_numpy_array(tour, name="tour")

    _validate_single_coords(coords)

    n_nodes = coords.shape[0]

    if validate:
        validation = validate_tour(tour, n_nodes)
        if not validation.is_valid:
            raise ValueError(f"Invalid tour: {validation.reason}")

    if tour.ndim != 1:
        raise ValueError(f"'tour' must be 1D, but got ndim={tour.ndim}.")
    if len(tour) != n_nodes:
        raise ValueError(f"'tour' length {len(tour)} does not match n_nodes={n_nodes}.")

    tour_int = tour.astype(np.int64, copy=False)
    ordered_coords = coords[tour_int]
    next_coords = np.roll(ordered_coords, shift=-1, axis=0)
    segment_lengths = np.linalg.norm(ordered_coords - next_coords, axis=1)

    return float(np.sum(segment_lengths))


def compute_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """
    Compute the full pairwise Euclidean distance matrix for a single instance.

    Args:
        coords:
            Array of shape (n_nodes, 2).

    Returns:
        Distance matrix of shape (n_nodes, n_nodes).
    """
    coords = _ensure_numpy_array(coords, name="coords").astype(np.float64, copy=False)
    _validate_single_coords(coords)

    diffs = coords[:, None, :] - coords[None, :, :]
    return np.linalg.norm(diffs, axis=-1)


def compute_tour_length_from_distance_matrix(distance_matrix: np.ndarray,tour: np.ndarray, *,
                                                    validate: bool = True) -> float:
    """
    Compute the total length of a tour using a precomputed distance matrix.

    Args:
        distance_matrix:
            Array of shape (n_nodes, n_nodes).
        tour:
            Array of shape (n_nodes,), representing a permutation of node indices.
        validate:
            If True, validate the tour before computing the length.

    Returns:
        Total closed-tour length.

    Raises:
        ValueError:
            If shapes are invalid or the tour is invalid when validate=True.
    """
    distance_matrix = _ensure_numpy_array(
        distance_matrix, name="distance_matrix"
    ).astype(np.float64, copy=False)
    tour = _ensure_numpy_array(tour, name="tour")

    if distance_matrix.ndim != 2:
        raise ValueError(
            "'distance_matrix' must have shape (n_nodes, n_nodes), "
            f"but got ndim={distance_matrix.ndim}."
        )

    n_rows, n_cols = distance_matrix.shape
    if n_rows != n_cols:
        raise ValueError(
            "'distance_matrix' must be square, "
            f"but got shape {distance_matrix.shape}."
        )
    if n_rows == 0:
        raise ValueError("'distance_matrix' must contain at least one node.")
    if not np.all(np.isfinite(distance_matrix)):
        raise ValueError("'distance_matrix' contains non-finite values.")

    if validate:
        validation = validate_tour(tour, n_rows)
        if not validation.is_valid:
            raise ValueError(f"Invalid tour: {validation.reason}")

    if tour.ndim != 1:
        raise ValueError(f"'tour' must be 1D, but got ndim={tour.ndim}.")
    if len(tour) != n_rows:
        raise ValueError(f"'tour' length {len(tour)} does not match n_nodes={n_rows}.")

    tour_int = tour.astype(np.int64, copy=False)
    next_tour = np.roll(tour_int, shift=-1)
    return float(np.sum(distance_matrix[tour_int, next_tour]))


def compute_batch_tour_lengths(coords_batch: np.ndarray, tours_batch: np.ndarray, *, 
                                    validate: bool = True, invalid_value: float = np.nan,) -> np.ndarray:
    """
    Compute TSP tour lengths for a batch of instances.

    Args:
        coords_batch:
            Array of shape (batch_size, n_nodes, 2).
        tours_batch:
            Array of shape (batch_size, n_nodes).
        validate:
            If True, invalid tours will not raise; instead they will be assigned
            `invalid_value`.
        invalid_value:
            Value assigned to invalid tours when validate=True.

    Returns:
        Array of shape (batch_size,) of tour lengths.
    """
    coords_batch = _ensure_numpy_array(coords_batch, name="coords_batch").astype(
        np.float64, copy=False
    )
    tours_batch = _ensure_numpy_array(tours_batch, name="tours_batch")

    _validate_batch_coords(coords_batch)

    if tours_batch.ndim != 2:
        raise ValueError(
            "'tours_batch' must have shape (batch_size, n_nodes), "
            f"but got ndim={tours_batch.ndim}."
        )

    batch_size, n_nodes, _ = coords_batch.shape
    if tours_batch.shape != (batch_size, n_nodes):
        raise ValueError(
            "'tours_batch' must have shape "
            f"({batch_size}, {n_nodes}), but got {tours_batch.shape}."
        )

    lengths = np.empty(batch_size, dtype=np.float64)

    for idx in range(batch_size):
        tour = tours_batch[idx]
        coords = coords_batch[idx]

        if validate:
            validation = validate_tour(tour, n_nodes)
            if not validation.is_valid:
                lengths[idx] = invalid_value
                continue

        lengths[idx] = compute_tour_length(coords, tour, validate=False)

    return lengths


def compute_validity_mask(tours_batch: np.ndarray, n_nodes: int) -> np.ndarray:
    """
    Compute a boolean mask indicating whether each batch tour is valid.

    Args:
        tours_batch:
            Array of shape (batch_size, n_nodes).
        n_nodes:
            Number of nodes expected in each instance.

    Returns:
        Boolean array of shape (batch_size,).
    """
    tours_batch = _ensure_numpy_array(tours_batch, name="tours_batch")

    if tours_batch.ndim != 2:
        raise ValueError(
            "'tours_batch' must have shape (batch_size, n_nodes), "
            f"but got ndim={tours_batch.ndim}."
        )

    return np.array(
        [validate_tour(tour, n_nodes).is_valid for tour in tours_batch],
        dtype=bool,
    )


def compute_validity_rate(tours_batch: np.ndarray, n_nodes: int) -> float:
    """
    Compute the fraction of valid tours in a batch.

    Args:
        tours_batch:
            Array of shape (batch_size, n_nodes).
        n_nodes:
            Number of nodes expected in each instance.

    Returns:
        Fraction of valid tours in [0, 1].
    """
    validity_mask = compute_validity_mask(tours_batch, n_nodes)
    return float(np.mean(validity_mask))


def compute_optimality_gap(predicted_cost: float, reference_cost: float) -> float:
    """
    Compute the relative optimality gap against a reference cost.

    The returned value is:

        (predicted_cost - reference_cost) / reference_cost

    Example:
        predicted_cost = 10.8, reference_cost = 10.0 -> gap = 0.08 (8%)

    Args:
        predicted_cost:
            Cost produced by the model or heuristic.
        reference_cost:
            Cost from an exact or strong baseline solver.

    Returns:
        Relative gap as a float.

    Raises:
        ValueError:
            If costs are non-finite or reference_cost is not strictly positive.
    """
    if not np.isfinite(predicted_cost):
        raise ValueError("'predicted_cost' must be finite.")
    if not np.isfinite(reference_cost):
        raise ValueError("'reference_cost' must be finite.")
    if reference_cost <= EPS:
        raise ValueError("'reference_cost' must be strictly positive.")

    return float((predicted_cost - reference_cost) / reference_cost)


def compute_batch_optimality_gaps(
    predicted_costs: np.ndarray,
    reference_costs: np.ndarray,
    *,
    skip_invalid: bool = True,
) -> np.ndarray:
    """
    Compute relative optimality gaps for a batch of instances.

    Args:
        predicted_costs:
            Array of shape (batch_size,).
        reference_costs:
            Array of shape (batch_size,).
        skip_invalid:
            If True, entries with invalid/non-finite predicted costs or
            non-positive reference costs are set to np.nan.
            If False, invalid entries raise ValueError.

    Returns:
        Array of shape (batch_size,) containing relative gaps.
    """
    predicted_costs = _ensure_numpy_array(
        predicted_costs, name="predicted_costs"
    ).astype(np.float64, copy=False)
    reference_costs = _ensure_numpy_array(
        reference_costs, name="reference_costs"
    ).astype(np.float64, copy=False)

    if predicted_costs.ndim != 1:
        raise ValueError(
            f"'predicted_costs' must be 1D, but got ndim={predicted_costs.ndim}."
        )
    if reference_costs.ndim != 1:
        raise ValueError(
            f"'reference_costs' must be 1D, but got ndim={reference_costs.ndim}."
        )
    if predicted_costs.shape != reference_costs.shape:
        raise ValueError(
            "'predicted_costs' and 'reference_costs' must have the same shape, "
            f"but got {predicted_costs.shape} and {reference_costs.shape}."
        )

    gaps = np.empty_like(predicted_costs, dtype=np.float64)

    for idx, (pred, ref) in enumerate(zip(predicted_costs, reference_costs)):
        is_valid_entry = np.isfinite(pred) and np.isfinite(ref) and ref > EPS
        if not is_valid_entry:
            if skip_invalid:
                gaps[idx] = np.nan
            else:
                raise ValueError(
                    f"Invalid batch entry at index {idx}: "
                    f"predicted_cost={pred}, reference_cost={ref}."
                )
            continue

        gaps[idx] = (pred - ref) / ref

    return gaps


def summarize_costs(
    costs: np.ndarray,
    *,
    prefix: str = "",
) -> dict[str, float]:
    """
    Compute descriptive summary statistics for a 1D array of costs.

    Non-finite values are ignored.

    Args:
        costs:
            Array-like of shape (n,).
        prefix:
            Optional prefix for output keys, e.g. "pred_" -> "pred_mean_cost".

    Returns:
        Dictionary containing count, mean, std, min, max, median, and percentiles.

    Raises:
        ValueError:
            If there are no finite values.
    """
    costs = _ensure_numpy_array(costs, name="costs").astype(np.float64, copy=False)

    if costs.ndim != 1:
        raise ValueError(f"'costs' must be 1D, but got ndim={costs.ndim}.")

    finite_costs = costs[np.isfinite(costs)]
    if finite_costs.size == 0:
        raise ValueError("No finite costs available to summarize.")

    return {
        f"{prefix}count": float(finite_costs.size),
        f"{prefix}mean_cost": float(np.mean(finite_costs)),
        f"{prefix}std_cost": float(np.std(finite_costs)),
        f"{prefix}min_cost": float(np.min(finite_costs)),
        f"{prefix}max_cost": float(np.max(finite_costs)),
        f"{prefix}median_cost": float(np.median(finite_costs)),
        f"{prefix}p05_cost": float(np.percentile(finite_costs, 5)),
        f"{prefix}p95_cost": float(np.percentile(finite_costs, 95)),
    }


def summarize_gaps(
    gaps: np.ndarray,
    *,
    prefix: str = "",
) -> dict[str, float]:
    """
    Compute descriptive summary statistics for a 1D array of optimality gaps.

    Non-finite values are ignored.

    Args:
        gaps:
            Array-like of shape (n,).
        prefix:
            Optional prefix for output keys, e.g. "val_" -> "val_mean_gap".

    Returns:
        Dictionary containing count, mean, std, min, max, median, and percentiles.

    Raises:
        ValueError:
            If there are no finite values.
    """
    gaps = _ensure_numpy_array(gaps, name="gaps").astype(np.float64, copy=False)

    if gaps.ndim != 1:
        raise ValueError(f"'gaps' must be 1D, but got ndim={gaps.ndim}.")

    finite_gaps = gaps[np.isfinite(gaps)]
    if finite_gaps.size == 0:
        raise ValueError("No finite gaps available to summarize.")

    return {
        f"{prefix}count": float(finite_gaps.size),
        f"{prefix}mean_gap": float(np.mean(finite_gaps)),
        f"{prefix}std_gap": float(np.std(finite_gaps)),
        f"{prefix}min_gap": float(np.min(finite_gaps)),
        f"{prefix}max_gap": float(np.max(finite_gaps)),
        f"{prefix}median_gap": float(np.median(finite_gaps)),
        f"{prefix}p05_gap": float(np.percentile(finite_gaps, 5)),
        f"{prefix}p95_gap": float(np.percentile(finite_gaps, 95)),
    }


def summarize_evaluation(
    coords_batch: np.ndarray,
    tours_batch: np.ndarray,
    reference_costs: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Compute a standard end-to-end evaluation summary for TSP predictions.

    This function:
    - validates tours
    - computes predicted tour lengths
    - reports validity rate
    - optionally computes and summarizes optimality gaps if reference costs are given

    Args:
        coords_batch:
            Array of shape (batch_size, n_nodes, 2).
        tours_batch:
            Array of shape (batch_size, n_nodes).
        reference_costs:
            Optional array of shape (batch_size,) of baseline/exact costs.

    Returns:
        Dictionary of scalar metrics suitable for logging.
    """
    coords_batch = _ensure_numpy_array(coords_batch, name="coords_batch")
    tours_batch = _ensure_numpy_array(tours_batch, name="tours_batch")

    _validate_batch_coords(coords_batch)

    if tours_batch.ndim != 2:
        raise ValueError(
            "'tours_batch' must have shape (batch_size, n_nodes), "
            f"but got ndim={tours_batch.ndim}."
        )

    batch_size, n_nodes, _ = coords_batch.shape
    if tours_batch.shape != (batch_size, n_nodes):
        raise ValueError(
            "'tours_batch' must have shape "
            f"({batch_size}, {n_nodes}), but got {tours_batch.shape}."
        )

    validity_mask = compute_validity_mask(tours_batch, n_nodes)
    predicted_costs = compute_batch_tour_lengths(
        coords_batch,
        tours_batch,
        validate=True,
        invalid_value=np.nan,
    )

    summary: dict[str, float] = {
        "num_instances": float(batch_size),
        "num_valid_tours": float(np.sum(validity_mask)),
        "validity_rate": float(np.mean(validity_mask)),
    }

    try:
        summary.update(summarize_costs(predicted_costs))
    except ValueError:
        summary.update(
            {
                "count": 0.0,
                "mean_cost": np.nan,
                "std_cost": np.nan,
                "min_cost": np.nan,
                "max_cost": np.nan,
                "median_cost": np.nan,
                "p05_cost": np.nan,
                "p95_cost": np.nan,
            }
        )

    if reference_costs is not None:
        reference_costs = _ensure_numpy_array(
            reference_costs, name="reference_costs"
        ).astype(np.float64, copy=False)

        if reference_costs.ndim != 1:
            raise ValueError(
                f"'reference_costs' must be 1D, but got ndim={reference_costs.ndim}."
            )
        if reference_costs.shape[0] != batch_size:
            raise ValueError(
                f"'reference_costs' must have length {batch_size}, "
                f"but got length {reference_costs.shape[0]}."
            )

        gaps = compute_batch_optimality_gaps(
            predicted_costs=predicted_costs,
            reference_costs=reference_costs,
            skip_invalid=True,
        )

        try:
            summary.update(summarize_gaps(gaps))
        except ValueError:
            summary.update(
                {
                    "mean_gap": np.nan,
                    "std_gap": np.nan,
                    "min_gap": np.nan,
                    "max_gap": np.nan,
                    "median_gap": np.nan,
                    "p05_gap": np.nan,
                    "p95_gap": np.nan,
                }
            )

    return summary


__all__ = [
    "TourValidationResult",
    "validate_tour",
    "is_valid_tour",
    "compute_tour_length",
    "compute_distance_matrix",
    "compute_tour_length_from_distance_matrix",
    "compute_batch_tour_lengths",
    "compute_validity_mask",
    "compute_validity_rate",
    "compute_optimality_gap",
    "compute_batch_optimality_gaps",
    "summarize_costs",
    "summarize_gaps",
    "summarize_evaluation",
]