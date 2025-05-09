""" Quantum-related utilities """
from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy import ndarray
from pysat.examples.hitman import Hitman


def get_different_inds(basis_1: Sequence, basis_2: Sequence, ignore_ind: int) -> list[int]:
    """ Returns different indices between two bases, ignoring ignore_ind. """
    return [ind for ind in range(len(basis_1)) if ind != ignore_ind and basis_1[ind] != basis_2[ind]]


def solve_minimum_hitting_set(sets: list[list[int]]) -> list[int]:
    """ Finds the smallest set of integers that overlaps with all given sets. """
    hitman = Hitman()
    for set in sets:
        hitman.hit(set)
    solution = hitman.get()
    return solution


def get_neighbor_ind(bases: ndarray, origin_ind: int, interaction_ind: int) -> int | None:
    """ Returns index of basis adjacent to origin_ind via interaction_ind or None if no such index exists. """
    origin = bases[origin_ind]
    destination = origin ^ (np.arange(len(origin)) == interaction_ind)
    destination_ind = np.where(np.all(destination == bases, axis=1))[0]
    if destination_ind:
        return destination_ind[0]
    return None


def find_min_control_set(existing_bases: ndarray, target_basis_ind: int, interaction_ind: int, multiedge: bool = False) -> list[int]:
    """
    Finds minimum set of controls necessary to distinguish basis given by target_basis_ind and its pair different in interaction_ind from existing_bases.
    :param existing_bases: 2D array of existing bases. Each row is a non-zero basis.
    :param target_basis_ind: Index of the target basis state in the existing_bases to distinguish from the rest of them.
    :param interaction_ind: Index that has to be excluded from consideration for the control set.
                            Neighbor of target_state in interaction_ind dimension is not distinguished from target_basis.
    :param multiedge: If True, all parallel walks with non-zero amplitudes on both ends are ignored when determining control set.
    :return: Minimum set of control indices necessary to select the target state.
    """
    ignore_rows = {target_basis_ind}
    destination_ind = get_neighbor_ind(existing_bases, target_basis_ind, interaction_ind)
    if destination_ind is not None:
        ignore_rows.add(destination_ind)

    if multiedge:
        for basis_ind, basis in enumerate(existing_bases):
            if basis_ind in ignore_rows:
                continue
            neighbor_ind = get_neighbor_ind(existing_bases, basis_ind, interaction_ind)
            if neighbor_ind is not None:
                ignore_rows.update((basis_ind, neighbor_ind))

    origin = existing_bases[target_basis_ind]
    different_inds = [get_different_inds(basis, origin, interaction_ind) for row_ind, basis in enumerate(existing_bases) if row_ind not in ignore_rows]
    return solve_minimum_hitting_set(different_inds)


def find_min_control_set_2(existing_states: list[list[int]] | ndarray, target_state_ind: int, candidate_interaction_inds: Sequence, multiedge: bool = False) -> \
        (ndarray, ndarray, int):
    """ Higher level wrapper around find_min_control_set_1. Handles state transformation under CX conjugation and iterates over candidate interaction indices to find the best.
    Returns control indices, values and selected interaction index. """
    result = None
    for interaction_ind in candidate_interaction_inds:
        transformed_states = np.array(existing_states)
        cx_target_inds = [ind for ind in candidate_interaction_inds if ind != interaction_ind]
        transformed_states[np.ix_(transformed_states[:, interaction_ind] == 1, cx_target_inds)] ^= 1
        control_inds = find_min_control_set(transformed_states, target_state_ind, interaction_ind, multiedge)
        if result is None or len(result[0]) > len(control_inds):
            control_vals = transformed_states[target_state_ind, control_inds]
            result = (np.array(control_inds), control_vals, interaction_ind)
    return result


def get_cx_cost_rx(num_controls: int) -> int:
    """ Returns the number of CX gates in the decomposition of multi-controlled Rx gate with the specified number of controls. """
    cx_by_num_controls = [0, 2, 8, 20, 24, 40, 56, 80, 104]
    if num_controls < len(cx_by_num_controls):
        return cx_by_num_controls[num_controls]
    else:
        return cx_by_num_controls[-1] + (num_controls - len(cx_by_num_controls) - 1) * 16


def get_hamming_distance(str1: str, str2: str) -> int:
    return sum(c1 != c2 for c1, c2 in zip(str1, str2))


def change_basis(basis: str, change_inds: list[int]) -> str:
    """ Flips bitstring in the specified indices. """
    changed_basis = np.array([int(val) for val in basis])
    changed_basis[change_inds] ^= 1
    changed_basis = "".join([str(val) for val in changed_basis])
    return changed_basis


def change_basis_if(basis: str, change_inds: list[int], control_ind: int) -> str:
    """ Flips bitstring in the specified indices if the control index is 1. """
    if basis[control_ind] == "0":
        return basis
    return change_basis(basis, change_inds)


def get_num_neighbors(bases: ndarray, basis_ind: int):
    """ Returns the number of rows in bases that are different from row with index basis_ind in exactly 1 position, i.e. number of neighbor bases. """
    target_basis = bases[basis_ind]
    return sum(np.sum(basis != target_basis) == 1 for basis in bases)
