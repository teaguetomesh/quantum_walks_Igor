from __future__ import annotations

import heapq
from abc import ABC, abstractmethod
from dataclasses import dataclass

import networkx as nx
import numpy as np
from networkx.classes import Graph
from numpy import ndarray
import numpy.random as random

from src.utilities import array_to_str, get_average_neighbors


class DensePermutationGenerator(ABC):
    """ Generates sparse to dense state permutations. """

    @abstractmethod
    def get_permutation(self, state: dict[str, complex]) -> (dict[str, str], list[int]):
        """ Generates sparse to dense state permutations for a given sparse state.
        Returns dict of old state -> new state mapping and indices of qubits forming the dense subspace.
        Permutation for the bases that have not been specified is arbitrary. """
        pass


class SequentialPermutator(DensePermutationGenerator):
    """ Generates dense permutation in sequentially increasing order. """

    def get_permutation(self, state: dict[str, complex]) -> (dict[str, str], list[int]):
        num_qubits_sparse = len(next(iter(state)))
        mapping = {basis: format(i, f'0{num_qubits_sparse}b') for i, basis in enumerate(state)}
        num_qubits_dense = int(np.ceil(np.log2(len(state))))
        qubits_dense = list(range(num_qubits_sparse))[-num_qubits_dense:]
        return mapping, qubits_dense


class MatchPermutator(DensePermutationGenerator):
    """ Generates dense permutation based on Hamming distance. """

    def get_permutation(self, state: dict[str, complex]) -> (dict[str, str], list[int]):
        num_qubits_sparse = len(next(iter(state)))
        num_qubits_dense = int(np.ceil(np.log2(len(state))))
        destinations = [format(i, f'0{num_qubits_sparse}b') for i in range(2 ** num_qubits_dense)]
        match_graph = Graph()
        for basis in state:
            for destination in destinations:
                distance = sum(b1 != b2 for b1, b2 in zip(basis, destination))
                match_graph.add_edge(basis + '_o', destination + '_d', weight=distance)
        pairs = list(nx.min_weight_matching(match_graph))
        for i in range(len(pairs)):
            if pairs[i][0][-1] == 'd':
                pairs[i] = pairs[i][::-1]
        permutation = {pair[0][:-2]: pair[1][:-2] for pair in pairs}
        qubits_dense = list(range(num_qubits_sparse))[-num_qubits_dense:]
        return permutation, qubits_dense


class HypercubePermutator(DensePermutationGenerator):
    """ Splits the target state into hypercubes until their permutation fits into a dense subspace. """

    @dataclass
    class Hypercube:
        """ Hypercube in the state space.
        :var coords: Hypercube coordinates. Could either be 0, 1 or -1 to show the spanned dimension.
        :var basis_inds: Indices of the states covered by this hypercube. """
        coords: ndarray
        basis_inds: set[int] = None

        def get_dimension_inds(self) -> list[int]:
            """ Returns indices of spanned dimensions. """
            return np.where(self.coords == -1)[0].tolist()

        def get_num_dimensions(self) -> int:
            """ Returns the number of dimensions in this hypercube. """
            return len(self.get_dimension_inds())

        def get_num_zero_amplitudes(self) -> int:
            """ Returns the number of zero amplitudes in this hypercube. """
            num_dims = self.get_num_dimensions()
            return 2 ** num_dims - len(self.basis_inds)

    @dataclass
    class SplitNode:
        """ Node in a splitting tree for a hierarchy of hypercube splits.
        :var hcube: Underlying hypercube.
        :var children: Children nodes. """
        hcube: Hypercube
        children: (SplitNode, SplitNode) = None

        def __lt__(self, other: SplitNode):
            return self.hcube.get_num_zero_amplitudes() > other.hcube.get_num_zero_amplitudes()

        def get_height(self):
            if self.children is None:
                return 1
            return 1 + max(self.children[0].get_height(), self.children[1].get_height())

    def select_best_hypercube(self, hcubes: list[Hypercube], bases: ndarray, target_coords: ndarray = None) -> Hypercube:
        """ Selects the best hypercube among the given list. """
        best_hcubes = np.array(hcubes)

        bases_covered = np.array([len(hcube.basis_inds) for hcube in hcubes])
        max_count = max(bases_covered)
        best_hcubes = best_hcubes[bases_covered == max_count]

        if len(best_hcubes) > 1:
            average_neighbors = []
            for hcube in best_hcubes:
                covered_bases = bases[list(hcube.basis_inds), :]
                average_neighbors.append(get_average_neighbors(covered_bases))
            max_neighbors = max(average_neighbors)
            best_hcubes = best_hcubes[np.array(average_neighbors) == max_neighbors]

        if len(best_hcubes) > 1 and target_coords is not None:
            target_distances = np.array([np.sum(hcube.coords[hcube.coords != -1] ^ target_coords) for hcube in best_hcubes])
            min_distance = min(target_distances)
            best_hcubes = best_hcubes[target_distances == min_distance]

        return random.choice(best_hcubes)

    def find_covering_hypercube(self, bases: ndarray, num_dims: int, uncovered_inds: list[int], allowed_dims: list[int], target_coords: ndarray = None) -> Hypercube:
        """ Finds a hypercube map that contains the largest number of amplitudes and has specified number of dimensions.
        allowed_dims specifies which dimensions could be considered for spanning. """
        fixed_dims = list(set(range(bases.shape[1])) - set(allowed_dims))
        unique_rows, inverse = np.unique(bases[np.ix_(uncovered_inds, fixed_dims)], axis=0, return_inverse=True)
        candidate_hcubes = []
        for i, row in enumerate(unique_rows):
            hcube_coords = -np.ones(bases.shape[1], dtype=int)
            hcube_coords[fixed_dims] = row
            covered_inds = set(np.array(uncovered_inds)[np.where(inverse == i)[0]])
            candidate_hcubes.append(self.Hypercube(hcube_coords, covered_inds))
        best_hcube = self.select_best_hypercube(candidate_hcubes, bases, target_coords)

        while best_hcube.get_num_dimensions() > num_dims:
            candidate_hcubes = []
            for dim in best_hcube.get_dimension_inds():
                for val in [0, 1]:
                    coords = best_hcube.coords.copy()
                    coords[dim] = val
                    basis_inds = {ind for ind in best_hcube.basis_inds if bases[ind, dim] == val}
                    candidate_hcubes.append(self.Hypercube(coords, basis_inds))
            best_hcube = self.select_best_hypercube(candidate_hcubes, bases, target_coords)
        return best_hcube

    def find_covering_hypercubes(self, bases: ndarray) -> SplitNode:
        """ Attempts to greedily find a set of hypercubes such that they cover all given bases, can be permuted into a dense subspace and their number is minimized.
        The set is given by leaves of a split tree that describes their permutation into the dense subspace and whose root is returned. """
        dense_num_dims = int(np.ceil(np.log2(bases.shape[0])))
        root = self.SplitNode(self.find_covering_hypercube(bases, dense_num_dims, list(range(bases.shape[0])), list(range(bases.shape[1]))))
        nodes = [root]
        free_amplitude_inds = set(range(bases.shape[0])) - root.hcube.basis_inds
        while len(free_amplitude_inds) > 0:
            parent_node = heapq.heappop(nodes)
            parent_dim_inds = parent_node.hcube.get_dimension_inds()
            next_amplitude_inds = set(parent_node.hcube.basis_inds)
            child_1 = self.find_covering_hypercube(bases, len(parent_dim_inds) - 1, list(next_amplitude_inds), parent_dim_inds)
            next_amplitude_inds -= child_1.basis_inds
            if len(next_amplitude_inds) == 0:
                next_amplitude_inds = free_amplitude_inds
            split_dim = next(iter(set(parent_node.hcube.get_dimension_inds()) - set(child_1.get_dimension_inds())))
            target_coords = child_1.coords.copy()
            target_coords[split_dim] = 1 - target_coords[split_dim]
            child_2 = self.find_covering_hypercube(bases, len(parent_dim_inds) - 1, list(next_amplitude_inds), child_1.get_dimension_inds(), target_coords[target_coords != -1])
            next_amplitude_inds -= child_2.basis_inds
            parent_node.children = (self.SplitNode(child_1), self.SplitNode(child_2))
            heapq.heappush(nodes, parent_node.children[0])
            heapq.heappush(nodes, parent_node.children[1])
        return root

    def extract_permutation(self, node: SplitNode, bases: ndarray, target_coords: ndarray) -> dict[str, str]:
        """ Builds state permutation for states from the given node into target_coords. """
        if node.children is None:
            permutation = {}
            for basis_ind in node.hcube.basis_inds:
                target = bases[basis_ind, :].copy()
                for i, val in enumerate(target_coords):
                    if val == -1:
                        continue
                    target[i] = val
                permutation[array_to_str(bases[basis_ind, :])] = array_to_str(target)
        else:
            split_dim = next(iter(set(node.hcube.get_dimension_inds()) - set(node.children[0].hcube.get_dimension_inds())))
            permutations = []
            remaining_vals = {0, 1}
            for child in node.children:
                child_target_coords = target_coords.copy()
                if len(remaining_vals) > 1:
                    child_target_coords[split_dim] = child.hcube.coords[split_dim]
                    remaining_vals.remove(child.hcube.coords[split_dim])
                else:
                    child_target_coords[split_dim] = next(iter(remaining_vals))
                permutations.append(self.extract_permutation(child, bases, child_target_coords))
            permutation = permutations[0] | permutations[1]
        return permutation

    def get_all_movable_leaves(self, root: SplitNode, covered_inds: set[int]) -> list[SplitNode]:
        if root.children is None:
            return [root] if next(iter(root.hcube.basis_inds)) not in covered_inds else []
        return self.get_all_movable_leaves(root.children[0], covered_inds) + self.get_all_movable_leaves(root.children[1], covered_inds)

    def get_permutation(self, state: dict[str, complex]) -> (dict[str, str], list[int]):
        all_bases = np.array([[int(val) for val in basis] for basis in state])
        root = self.find_covering_hypercubes(all_bases)
        permutation = self.extract_permutation(root, all_bases, root.hcube.coords)
        root_dims = root.hcube.get_dimension_inds()
        return permutation, root_dims
