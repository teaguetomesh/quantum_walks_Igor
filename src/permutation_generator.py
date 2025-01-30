from abc import ABC, abstractmethod

import networkx as nx
import numpy as np
from networkx.classes import Graph, DiGraph


class PermutationGenerator(ABC):
    """ Generates state permutations. """

    @abstractmethod
    def get_permutation(self, state: dict[str, complex]) -> dict[str, str]:
        """ Generates permutation for a given state. Returns dict of old state -> new state mapping. Permutation for the bases that have not been specified is arbitrary. """
        pass


class SequentialPermutator(PermutationGenerator):
    """ Generates dense permutation in sequentially increasing order. """

    def get_permutation(self, state: dict[str, complex]) -> dict[str, str]:
        num_qubits = len(next(iter(state)))
        return {basis: format(i, f'0{num_qubits}b') for i, basis in enumerate(state)}


class MatchPermutator(PermutationGenerator):
    """ Generates dense permutation based on Hamming distance. """

    def get_permutation(self, state: dict[str, complex]) -> dict[str, str]:
        num_qubits = len(next(iter(state)))
        num_qubits_dense = int(np.ceil(np.log2(len(state))))
        destinations = [format(i, f'0{num_qubits}b') for i in range(2 ** num_qubits_dense)]
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
        return permutation
