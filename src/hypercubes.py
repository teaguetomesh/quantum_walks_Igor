import copy
import itertools as it
from dataclasses import dataclass

import exact_cover as ec
import numpy as np
from numpy import ndarray


def find_all_dense_hypercubes(bases: ndarray) -> list[Hypercube]:
    """ Finds all hypercubes that given bases can form. """
    all_hcubes = [Hypercube(row, {i}) for i, row in enumerate(bases)]
    last_group = all_hcubes
    next_group = []
    while len(last_group) > 1:
        for pair in it.combinations(last_group, 2):
            different_dimensions = pair[0].coords != pair[1].coords
            if sum(different_dimensions) == 1:
                hcube_merged = copy.deepcopy(pair[0])
                dim = np.where(different_dimensions)[0][0]
                hcube_merged.coords[dim] = -1
                hcube_merged.basis_inds |= pair[1].basis_inds
                next_group.append(hcube_merged)
        all_hcubes.extend(next_group)
        last_group = next_group
        next_group = []
    return all_hcubes


def build_cover_matrix(hcubes: list[Hypercube], total_bases: int) -> ndarray:
    """ Converts a list of hypercubes into cover matrix, i.e. boolean matrix where elem [i, j] is true if hypercube i covers basis j and false otherwise. """
    cover_matrix = np.zeros((len(hcubes), total_bases), dtype=bool)
    for i, hcube in enumerate(hcubes):
        cover_matrix[i, list(hcube.basis_inds)] = True
    return cover_matrix


def find_dense_covering_hypercubes(all_bases: ndarray) -> list[Hypercube]:
    """ Finds the smallest list of dense hypercubes that can cover all given bases. """
    all_hcubes = find_all_dense_hypercubes(all_bases)
    cover_matrix = build_cover_matrix(all_hcubes, all_bases.shape[0])
    all_solutions = ec.get_all_solutions(cover_matrix)
    smallest_solution = min(all_solutions, key=len)
    return [all_hcubes[i] for i in smallest_solution]
