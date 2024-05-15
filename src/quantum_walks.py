""" Functions that generate quantum walks. """
from abc import ABC, abstractmethod
from dataclasses import dataclass

import networkx as nx
import numpy as np
from networkx import Graph


@dataclass
class PathSegment:
    """
    Class that stores information about a particular segment in a state preparation path. Incorporates both phase and amplitude walks.
    :var labels: Initial and final basis state labels for this segment.
    :var phase_time: Time of the phase walk.
    :var amplitude_time: Time of the amplitude walk.
    """
    labels: tuple[str, str]
    phase_time: float
    amplitude_time: float


@dataclass
class PathFinder(ABC):
    """ Base class for implementations of particular traversal orders to prepare a given state. """

    @abstractmethod
    def build_travel_graph(self, bases: list[str]) -> Graph:
        """
        Builds a graph that describes connections between the bases during state preparation. Graph's "start" attribute has to be set to the starting basis of the path.
        :param bases: List of non-zero amplitudes in the target state.
        :return: Basis connectivity graph.
        """
        pass

    def set_graph_attributes(self, graph: Graph, target_state: dict[str, complex]):
        """
        Assigns necessary attributes of the travel graph.
        :param graph: Travel graph for state preparation.
        :param target_state: Target state for state preparation.
        """
        get_attributes = lambda label: {"current_phase": 1,
                                        "target_phase": target_state[label] / abs(target_state[label]),
                                        "current_prob": 0,
                                        "target_prob": abs(target_state[label]) ** 2}
        attributes = {key: get_attributes(key) for key in target_state.keys()}
        nx.set_node_attributes(graph, attributes)
        graph.nodes[graph.graph["start"]]["current_prob"] = 1
        bfs_edges = list(nx.bfs_edges(graph, graph.graph["start"]))[::-1]
        for edge in bfs_edges:
            graph.nodes[edge[0]]["target_prob"] += graph.nodes[edge[1]]["target_prob"]

    @staticmethod
    def find_zero_amplitude(target_state: dict[str, complex]) -> str | None:
        """
        Finds a basis state that is not a part of the target state.
        :param target_state: List of basis states with non-zero amplitude in the target state.
        :return: Any basis state that has zero amplitude in the target state.
        """
        num_qubits = len(next(iter(target_state.keys())))
        for i in range(2 ** num_qubits):
            basis_label = format(i, f"0{num_qubits}b")
            if basis_label not in target_state:
                return basis_label
        return None

    def get_path_segments(self, graph: Graph, zero_amplitude_state: str) -> list[PathSegment]:
        """
        Returns a list of path segments describing state preparation path.
        :param graph: Travel graph.
        :param zero_amplitude_state: Label of a basis state that is not a part of the target state.
        :return: List of path segments.
        """
        tol = 1e-10
        bfs_edges = list(nx.bfs_edges(graph, graph.graph["start"]))
        path = []
        for edge in bfs_edges:
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
                path.append(PathSegment((edge[1], zero_amplitude_state), phase_walk_time, 0))
        return path

    def get_path(self, target_state: dict[str, complex]) -> list[PathSegment]:
        """
        Returns state preparation path described by quantum walks.
        :param target_state: Target state to prepare.
        :return: List of path segments.
        """
        travel_graph = self.build_travel_graph(list(target_state.keys()))
        self.set_graph_attributes(travel_graph, target_state)
        zero_amplitude_label = self.find_zero_amplitude(target_state)
        return self.get_path_segments(travel_graph, zero_amplitude_label)


class PathFinderRandom(PathFinder):
    """ Connects the states via a random tree. """
    def build_travel_graph(self, bases: list[str]) -> Graph:
        random_tree = nx.generators.random_tree(len(bases))
        random_tree = nx.relabel_nodes(random_tree, {i: bases[i] for i in range(len(bases))})
        random_tree.graph["start"] = bases[0]
        return random_tree


class PathFinderLinear(PathFinder):
    """ Goes through the states in the same order they are listed in. """
    def build_travel_graph(self, bases: list[str]) -> Graph:
        graph = Graph()
        for i in range(len(bases) - 1):
            graph.add_edge(bases[i], bases[i + 1])
        graph.graph["start"] = bases[0]
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
        mst = nx.minimum_spanning_tree(distance_graph, algorithm='prim')
        mst.graph["start"] = bases[0]
        return mst
