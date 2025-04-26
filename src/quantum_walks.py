""" Functions that generate quantum walks. """
import copy
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import combinations

import graycode
import networkx as nx
import numpy as np
from networkx import Graph

from src.utilities.quantum import get_different_inds, solve_minimum_hitting_set, get_hamming_distance


@dataclass
class PathSegment:
    """
    Class that stores information about a particular segment in a state preparation path. Incorporates both phase and amplitude walks.
    :var labels: Initial and final basis state labels for this segment.
    :var phase_time: Time of the phase walk.
    :var amplitude_time: Time of the amplitude walk.
    :var interaction_index: Qubit index that should be used for the walks.
    :var phase_time_2: For terminal walks, time for extra leaf phase walk.
    """
    labels: (str, str)
    phase_time: float
    amplitude_time: float
    interaction_index: int = None
    phase_time_2: float = None


@dataclass
class PathFinder(ABC):
    """ Base class for implementations of particular traversal orders to prepare a given state. """

    @abstractmethod
    def build_travel_graph(self, bases: list[str]) -> Graph | tuple[Graph, list[(str, str)]]:
        """
        Builds a graph that describes connections between the bases during state preparation. Graph's "start" attribute has to be set to the starting basis of the path.
        :param bases: List of non-zero amplitudes in the target state.
        :return: 1) Basis connectivity graph. 2) Optionally, edge traversal order. Default BFS will be used if order is not provided.
        """
        pass

    @staticmethod
    def set_graph_attributes(graph: Graph, target_state: dict[str, complex], edge_order: list[(str, str)]):
        """
        Assigns necessary attributes of the travel graph.
        :param graph: Travel graph for state preparation.
        :param target_state: Target state for state preparation.
        :param edge_order: Custom edge order. Uses BFS by default.
        """
        get_attributes = lambda label: {"current_phase": 1,
                                        "target_phase": target_state[label] / abs(target_state[label]),
                                        "current_prob": 0,
                                        "target_prob": abs(target_state[label]) ** 2}
        attributes = {key: get_attributes(key) for key in target_state.keys()}
        nx.set_node_attributes(graph, attributes)
        graph.nodes[graph.graph["start"]]["current_prob"] = 1
        for edge in edge_order:
            graph.nodes[edge[0]]["target_prob"] += graph.nodes[edge[1]]["target_prob"]

    @staticmethod
    def get_hamming_generator(target_str: str) -> Iterator[str]:
        """
        Generates bit strings in the order of the Hamming distance to the target bit string.
        :param target_str: Target bit string.
        :return: Bit string generator.
        """
        all_inds = list(range(len(target_str)))
        for current_distance in range(1, len(target_str) + 1):
            for combo in combinations(all_inds, current_distance):
                next_str = np.array(list(map(int, target_str)))
                next_str[combo] = 1 - next_str[combo]
                next_str = "".join(map(str, next_str))
                yield next_str
        raise StopIteration

    @staticmethod
    def find_closest_zero_amplitude(target_state: dict[str, complex], target_basis: str) -> str | None:
        """
        Finds a basis state that is not a part of the target state.
        :param target_state: Dict of basis states with non-zero amplitude in the target state.
        :param target_basis: Target basis state around which the zero amplitude state will be searched.
        :return: Closest to the target basis state with zero amplitude.
        """
        for state in PathFinder.get_hamming_generator(target_basis):
            if state not in target_state:
                return state
        return None

    @staticmethod
    def get_path_segments(graph: Graph, target_state: dict[str, complex], edge_order: list[(str, str)]) -> list[PathSegment]:
        """
        Returns a list of path segments describing state preparation path.
        :param graph: Travel graph.
        :param target_state: Dict of non-zero amplitudes in the target state.
        :param edge_order: Custom edge order. Uses BFS by default.
        :return: List of path segments.
        """
        tol = 1e-10
        path = []
        for edge in edge_order:
            phase_walk_time = (1j * np.log(graph.nodes[edge[0]]["target_phase"] / graph.nodes[edge[0]]["current_phase"])).real
            graph.nodes[edge[0]]["current_phase"] = graph.nodes[edge[0]]["target_phase"]
            if abs(phase_walk_time) < tol:
                phase_walk_time = 0

            amplitude_walk_time = np.arcsin(np.sqrt(graph.nodes[edge[1]]["target_prob"] / graph.nodes[edge[0]]["current_prob"]))
            graph.nodes[edge[0]]["current_prob"] -= graph.nodes[edge[1]]["target_prob"]
            graph.nodes[edge[1]]["current_prob"] = graph.nodes[edge[1]]["target_prob"]
            graph.nodes[edge[1]]["current_phase"] = -1j * graph.nodes[edge[0]]["current_phase"]
            path.append(PathSegment(edge, phase_walk_time, amplitude_walk_time))

            if graph.degree(edge[1]) == 1:
                phase_walk_time = (1j * np.log(graph.nodes[edge[1]]["target_phase"] / graph.nodes[edge[1]]["current_phase"])).real
                graph.nodes[edge[1]]["current_phase"] = graph.nodes[edge[1]]["target_phase"]
                closest_zero_state = PathFinder.find_closest_zero_amplitude(target_state, edge[1])
                path.append(PathSegment((edge[1], closest_zero_state), phase_walk_time, 0))
        return path

    def get_path(self, target_state: dict[str, complex]) -> list[PathSegment]:
        """
        Returns state preparation path described by quantum walks.
        :param target_state: Target state to prepare.
        :return: List of path segments.
        """
        result = self.build_travel_graph(list(target_state.keys()))
        if isinstance(result, tuple):
            travel_graph, edge_order = result
        else:
            travel_graph = result
            edge_order = list(nx.bfs_edges(travel_graph, travel_graph.graph["start"]))

        self.set_graph_attributes(travel_graph, target_state, edge_order)
        return self.get_path_segments(travel_graph, target_state, edge_order)


class PathFinderRandom(PathFinder):
    """ Connects the states via a random tree. """
    def build_travel_graph(self, bases: list[str]) -> Graph:
        random_tree = nx.generators.random_tree(len(bases))
        random_tree = nx.relabel_nodes(random_tree, {i: bases[i] for i in range(len(bases))})
        random_tree.graph["start"] = bases[0]
        return random_tree


@dataclass
class PathFinderLinear(PathFinder):
    """ Goes through the states in the same order they are listed in. """
    order: Sequence[int] = None

    def build_travel_graph(self, bases: list[str]) -> Graph:
        graph = Graph()
        if self.order is not None:
            bases = np.array(bases)[self.order]
        for i in range(len(bases) - 1):
            graph.add_edge(bases[i], bases[i + 1])
        graph.graph["start"] = bases[0]
        return graph


class PathFinderGrayCode(PathFinder):
    """ Goes through the states in the order of Gray code. """
    def build_travel_graph(self, bases: list[str]) -> Graph:
        graph = Graph()
        gray_code = graycode.gen_gray_codes(len(bases[0]))
        gray_code_str = [f"{code:0{len(bases[0])}b}" for code in gray_code]
        last_basis = None
        for code in gray_code_str:
            if code in bases:
                if last_basis is None:
                    graph.graph["start"] = code
                else:
                    graph.add_edge(last_basis, code)
                last_basis = code
        return graph


def build_distance_graph(bases: list[str]) -> Graph:
    """
    Builds the fully connected graph on the nodes in bases where each weight is given by hamming distance.
    :param bases: List of bases to include in the graph.
    :return: Graph.
    """
    get_hamming_distance = lambda str1, str2: sum(c1 != c2 for c1, c2 in zip(str1, str2))
    graph = Graph()
    for i in range(len(bases)):
        for j in range(i + 1, len(bases)):
            graph.add_edge(bases[i], bases[j], weight=get_hamming_distance(bases[i], bases[j]))
    return graph


class PathFinderSHP(PathFinder):
    """ Returns the Shortest Hamiltonian Path throughout the target basis states. """
    def build_travel_graph(self, bases: list[str]) -> Graph:
        distance_graph = build_distance_graph(bases)
        shp_nodes = nx.approximation.traveling_salesman_problem(distance_graph, cycle=False)
        travel_graph = Graph()
        travel_graph.graph["start"] = shp_nodes[0]
        for i in range(len(shp_nodes) - 1):
            travel_graph.add_edge(shp_nodes[i], shp_nodes[i + 1])
        return travel_graph


class PathFinderMST(PathFinder):
    """ Returns the Minimum Spanning Tree on the basis states. """
    def build_travel_graph(self, bases: list[str]) -> Graph:
        distance_graph = build_distance_graph(bases)
        mst = nx.minimum_spanning_tree(distance_graph, algorithm="prim")
        mst.graph["start"] = bases[0]
        return mst


class PathFinderMHSNonlinear(PathFinder):
    """ The Minimum Hitting Set method on the basis states. The returned set of walks may be nonlinear. """

    @staticmethod
    def _get_single_hit_z2s(interaction_ind: int, remaining_basis: list[str], z1_diffs: list[list[int]], z1_mhs: list[int]) -> list[str]:
        """ Filters remaining_basis such that z1_diffs[i] intersects z1_mhs[i] only at interaction_ind. """
        new_remaining_basis = [remaining_basis[idx] for idx, elem in enumerate(z1_diffs) if set(elem).intersection(set(z1_mhs)) == {interaction_ind}]
        return new_remaining_basis

    def _get_z2_search(self, elem: str, basis: list[str]) -> (list[str], int):
        """ Returns the z2 search space and target qubit.
        :param elem: z1
        :param basis: all the basis states including elem.
        :return: z2 and interaction index """
        remaining_basis = [elem2 for elem2 in basis if elem2 != elem]
        diffs = [get_different_inds(elem, z2, -1) for z2 in remaining_basis]
        # Search for the target qubit.
        mhs = solve_minimum_hitting_set(diffs)
        # todo: the frequency should be counted over blocks that intersect the mhs at a single element.
        interaction_ind = min(mhs, key=lambda idx: sum([1 for block in diffs if idx in block]))
        z2_search = self._get_single_hit_z2s(interaction_ind, remaining_basis, diffs, mhs)
        return z2_search, interaction_ind

    def _select_z2(self, elem: str, z2_search: list[str]) -> (str, int):
        """ Returns a tuple of z2 and the number of controls required to differentiate z2 from the rest of the elements. """
        if len(z2_search) == 1:
            return z2_search[0], 0

        z2_mhs_scores = []
        for ind1 in range(len(z2_search)):
            diff_inds = [get_different_inds(z2_search[ind1], z2_search[ind2], -1) for ind2 in range(len(z2_search)) if ind2 != ind1]
            mhs = solve_minimum_hitting_set(diff_inds)
            z2_mhs_scores.append(len(mhs))
        z2, z2_score = min(zip(z2_search, z2_mhs_scores), key=lambda x: (x[1], get_hamming_distance(elem, x[0])))
        return z2, z2_score

    def order_states_mhs_z1(self, basis: list[str]) -> (str, str, int):
        """ Returns z1, z2, target. """
        diffs_z1 = [[get_different_inds(basis[z1_ind], basis[z2_ind], -1) for z2_ind in range(len(basis)) if z2_ind != z1_ind] for z1_ind in range(len(basis))]
        mhs_scores_z1 = [len(solve_minimum_hitting_set(elem)) for elem in diffs_z1]
        # gets a list of the tuples. First element in the tuple is a list of the possible z2s. The second element is the corresponding target.
        z2_search_spaces, targets = zip(*[self._get_z2_search(elem, basis) for elem in basis])
        best_z2s, best_z2_scores = zip(*[self._select_z2(elem, z2_search) for elem, z2_search in zip(basis, z2_search_spaces)])
        z1, z2, target = min(zip(basis, best_z2s, targets, mhs_scores_z1, best_z2_scores, diffs_z1),
                             key=lambda elem: (elem[3] + elem[4], -sum(len(block) for block in elem[5])))[:3]
        return z1, z2, target

    def update_nodes(self, z1: str, z2: str, visited: list[str], interaction_ind: int) -> list[str]:
        """ Updates visited. """
        diff_inds = get_different_inds(z1, z2, interaction_ind)
        visited = np.array([[int(char) for char in basis] for basis in visited])
        visited[np.ix_(visited[:, interaction_ind] == 1, diff_inds)] ^= 1
        visited = ["".join(map(str, elem)) for elem in visited]
        return visited

    def build_travel_graph(self, states: list[str]) -> (Graph, list[(str, str, int)]):
        graph = Graph()
        edge_order = []
        basis_original = copy.deepcopy(states)
        basis_mutable = copy.deepcopy(states)
        for _ in range(len(states) - 1):
            z1, z2, interaction_ind = self.order_states_mhs_z1(basis_mutable)
            z1_idx = basis_mutable.index(z1)
            z1_original = basis_original[z1_idx]
            z2_idx = basis_mutable.index(z2)
            z2_original = basis_original[z2_idx]
            edge_order.append((z1_original, z2_original, interaction_ind))
            basis_mutable.pop(z2_idx)
            basis_original.pop(z2_idx)
            basis_mutable = self.update_nodes(z1, z2, basis_mutable, interaction_ind)

        edge_order = edge_order[::-1]  # we worked backwards.
        for edge in edge_order:
            graph.add_edge(edge[0], edge[1])
        graph.graph["start"] = edge_order[0][0]
        return graph, edge_order

    def get_path_segments(self, graph: Graph, target_state: dict[str, complex], edge_order: list[(str, str, int)]) -> list[PathSegment]:
        tol = 1e-10
        path = []
        for edge in edge_order:
            interaction_ind = edge[2]
            edge = edge[:2]
            phase_walk_time = (1j * np.log(graph.nodes[edge[0]]["target_phase"] / graph.nodes[edge[0]]["current_phase"])).real
            graph.nodes[edge[0]]["current_phase"] = graph.nodes[edge[0]]["target_phase"]
            if abs(phase_walk_time) < tol:
                phase_walk_time = 0

            amplitude_walk_time = np.arcsin(np.sqrt(graph.nodes[edge[1]]["target_prob"] / graph.nodes[edge[0]]["current_prob"]))
            graph.nodes[edge[0]]["current_prob"] -= graph.nodes[edge[1]]["target_prob"]
            graph.nodes[edge[1]]["current_prob"] = graph.nodes[edge[1]]["target_prob"]
            graph.nodes[edge[1]]["current_phase"] = -1j * graph.nodes[edge[0]]["current_phase"]
            if graph.degree(edge[1]) > 1:
                path.append(PathSegment(list(edge), phase_walk_time, amplitude_walk_time, interaction_ind))
            else:
                phase_walk_time2 = (-1j * np.log(graph.nodes[edge[1]]["target_phase"] / graph.nodes[edge[1]]["current_phase"])).real
                graph.nodes[edge[1]]["current_phase"] = graph.nodes[edge[1]]["target_phase"]
                path.append(PathSegment(list(edge), -phase_walk_time, amplitude_walk_time, interaction_ind, phase_walk_time2))
        return path

    def get_path(self, target_state: dict[str, complex]) -> list[PathSegment]:
        travel_graph, edge_order = self.build_travel_graph(list(target_state.keys()))
        self.set_graph_attributes(travel_graph, target_state, edge_order[::-1])
        return self.get_path_segments(travel_graph, target_state, edge_order)
