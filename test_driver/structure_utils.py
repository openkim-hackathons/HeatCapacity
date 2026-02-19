"""Utility functions for structure manipulation and supercell generation."""

from typing import Tuple, Optional

import numpy as np
from ase import Atoms


def compute_supercell_reps_for_cutoff(cell: np.ndarray, r: float) -> Tuple[int, int, int]:
    """Compute supercell repetitions to ensure minimum image distance >= 2*r.

    This creates a supercell where each lattice direction has enough repetitions
    to ensure the minimum image distance is at least 2*r (i.e., the supercell
    can contain a sphere of radius r without self-interaction across periodic
    boundaries). This is particularly useful for non-cubic cells where uniform
    expansion would be suboptimal.

    Args:
        cell: 3x3 lattice vectors of the primitive cell (Å)
        r: Target radius in Å

    Returns:
        Tuple of repetitions (n_a, n_b, n_c) along each lattice vector
    """
    cell = np.asarray(cell)
    a, b, c = cell[0], cell[1], cell[2]

    # Volume of parallelepiped: v_p = |a · (b × c)|
    v_p = abs(np.dot(a, np.cross(b, c)))

    # Heights (perpendicular distances from origin to opposite face)
    # h_a = v_p / |b × c| (height along a direction)
    h_a = v_p / np.linalg.norm(np.cross(b, c))
    h_b = v_p / np.linalg.norm(np.cross(a, c))
    h_c = v_p / np.linalg.norm(np.cross(a, b))

    # Number of repeats needed: n = ceil(2*r / h)
    n_a = max(1, int(np.ceil(2 * r / h_a)))
    n_b = max(1, int(np.ceil(2 * r / h_b)))
    n_c = max(1, int(np.ceil(2 * r / h_c)))

    return (n_a, n_b, n_c)


def compute_supercell_reps_uniform_cubic(
    n_atoms_primitive: int, target_atoms: int
) -> Tuple[int, int, int]:
    """Compute uniform cubic supercell repetitions to reach a target atom count.

    Calculates the smallest integer `n` such that a supercell of `(n, n, n)`
    repetitions contains at least `target_atoms`. This method assumes a cubic
    expansion.

    Args:
        n_atoms_primitive: Number of atoms in the primitive unit cell.
        target_atoms: The desired minimum number of atoms in the supercell.

    Returns:
        Tuple of repetitions (n, n, n) for uniform cubic expansion.
    """
    n = max(1, int(np.ceil(np.cbrt(target_atoms / n_atoms_primitive))))
    return (n, n, n)


def compute_supercell_for_target_size(
    atoms: Atoms,
    target_size: int = 10000,
    current_radius: float = 20.0,
    previous_radius: Optional[float] = None,
    radius_step: float = 0.1,
    min_radius: float = 1.0
) -> Tuple[Atoms, Tuple[int, int, int]]:
    """Compute supercell using cutoff-based approach with target size constraint.

    Creates a supercell by iteratively adjusting the cutoff radius until the
    resulting supercell has the minimum number of atoms that is still greater
    than the target size.

    This approach is particularly useful for non-cubic cells where uniform
    expansion would be suboptimal, but you still want to control the total
    number of atoms.

    Args:
        atoms: Primitive unit cell as ASE Atoms object.
        target_size: Maximum number of atoms in the supercell. Default: 10000.
        current_radius: Starting cutoff radius in Å. Default: 20.0.
        previous_radius: Cutoff radius in Å of the previous iteration. Optional. Default: None.
        radius_step: Amount to decrease radius by in each iteration (Å). Default: 0.5.
        min_radius: Minimum radius to try before giving up (Å). Default: 1.0.

    Returns:
        Tuple of (supercell Atoms object, repeat tuple (n_a, n_b, n_c)).
        The supercell has natoms >= target_size.

    Raises:
        ValueError: If unable to find a supercell below target_size even at min_radius.
    """
    # If we've hit the minimum radius, raise an error
    if current_radius <= min_radius:
        raise ValueError(
            f"Unable to find supercell below target_size={target_size} atoms. "
            f"Even at min_radius={min_radius}Å, got {len(atoms)} atoms."
        )

    # Compute repetitions for current/new radius
    current_repeat = compute_supercell_reps_for_cutoff(atoms.get_cell(), current_radius)

    # Create new supercell
    current_supercell = atoms.repeat(current_repeat)
    current_natoms = len(current_supercell)

    if previous_radius is not None:
        # Compute repetitions for previous radius
        previous_repeat = compute_supercell_reps_for_cutoff(atoms.get_cell(), previous_radius)

        # Re-create previous supercell
        previous_supercell = atoms.repeat(previous_repeat)
        previous_natoms = len(previous_supercell)

        # Radius decreased
        if current_radius < previous_radius:

            if current_natoms < target_size:
                # Increase radius and try again
                new_radius = max(min_radius, current_radius + radius_step)
                return compute_supercell_for_target_size(
                    atoms, target_size, new_radius, current_radius, radius_step, min_radius
                )

            elif current_natoms >= target_size:
                # Decrease radius and try again
                new_radius = max(min_radius, current_radius - radius_step)
                return compute_supercell_for_target_size(
                    atoms, target_size, new_radius, current_radius, radius_step, min_radius
                )

        # Radius increased
        if current_radius > previous_radius:

            if current_natoms < target_size:
                # Increase radius and try again
                new_radius = max(min_radius, current_radius + radius_step)
                return compute_supercell_for_target_size(
                    atoms, target_size, new_radius, current_radius, radius_step, min_radius
                )

            elif current_natoms >= target_size:
                # Return previous supercell and repeat
                return current_supercell, current_repeat

    else:

        # As a first step, decrease the radius
        new_radius = max(min_radius, current_radius - radius_step)
        return compute_supercell_for_target_size(
            atoms, target_size, new_radius, current_radius, radius_step, min_radius
        )
