""" Shared functions. """
from __future__ import annotations

import ast
import inspect
import itertools
import textwrap
import traceback
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy import ndarray
from scipy import stats


def get_error_margin(data: Sequence, confidence: float = 0.95) -> float:
    z_val = stats.norm.ppf((1 + confidence) / 2)
    return z_val * np.std(data, ddof=1) / len(data) ** 0.5


def get_average_neighbors(bases: ndarray) -> float:
    """ Returns average number of neighbor bases in the given array of bases. """
    if bases.shape[0] == 0:
        return 0

    neighbors = 0
    for basis1, basis2 in itertools.combinations(bases, 2):
        if np.sum(basis1 != basis2) == 1:
            neighbors += 2
    return neighbors / bases.shape[0]


@dataclass
class GreedyNode:
    groups: list[int]
    items: list[int]
    score: int | float
    extra_output: tuple = None
    parent: GreedyNode | None = None
    children: list[GreedyNode] | None = None


def greedy_decision_tree(group_sizes: list[int], target_func: callable, ordered: bool = True, num_levels_ahead: int = 1, stop_no_improvement: bool = False, max_items: int = None) \
        -> list[GreedyNode]:
    """ Accepts a list of group sizes and a function that can be evaluated on a sequence of items from these groups.
    Finds the sequence that maximizes the value of the function by greedily adding items to the sequence.
    :param group_sizes: The list specifying the size of each group. Only one item can be chosen from each group and each group can only be chosen once.
    :param target_func: Target function to maximize.
    The function should accept 2 lists, where the 1st list contains selected group indices and the 2nd list contains corresponding item indices within each group.
    The function should return the score corresponding to given sequence. The sequence will be greedily adjusted to maximize this score.
    :param ordered: True if the input sequence is ordered, False otherwise.
    :param num_levels_ahead: Number of tree levels that will be fully explored before making next decision.
    :param stop_no_improvement: If True, stops adding items to the sequence if the last addition did not improve function's value.
    :param max_items: Maximum number of items that can be included in the sequence. If None, then keeps adding items until allowed_items are exhausted.
    :return: A list of equivalent nodes in the last layer of the decision tree. """
    def calculate_next_layer(last_layer):
        next_layer = []
        for node in last_layer:
            remaining_group_inds = set(range(len(group_sizes))) - set(node.groups)
            for group_ind in remaining_group_inds:
                if not ordered and group_ind < node.groups[-1]:
                    continue
                next_groups = node.groups + [group_ind]
                for item_ind in range(group_sizes[group_ind]):
                    next_items = node.items + [item_ind]
                    output = target_func(next_groups, next_items)
                    next_node = GreedyNode(next_groups, next_items, output[0], output[1:], node, [])
                    node.children.append(next_node)
                    next_layer.append(next_node)
        return next_layer

    if max_items is None:
        max_items = len(group_sizes)
    output = target_func([], [])
    last_layer = [GreedyNode([], [], output[0], output[1:], None, [])]
    while len(last_layer[0].groups) < min(num_levels_ahead, max_items):
        last_layer = calculate_next_layer(last_layer)

    best_node = None
    while True:
        prev_best_node = best_node
        best_node = max(last_layer, key=lambda x: x.score)
        if len(best_node.groups) == max_items or stop_no_improvement and prev_best_node is not None and best_node.score <= prev_best_node.score:
            break
        last_layer = best_node.parent.children
        last_layer = calculate_next_layer(last_layer)
    return [node for node in last_layer if node.score == best_node.score]


def array_to_str(arr: Sequence) -> str:
    """ Converts a given sequence to a string. """
    return ''.join([str(val) for val in arr])


def make_dict(*args):
    """ Creates a dictionary out of given arguments, using variable names passed to the call as keys. """
    current_frame = inspect.currentframe()
    func_name = current_frame.f_code.co_name
    caller_frame = current_frame.f_back
    caller_start_line_num = caller_frame.f_code.co_firstlineno
    call_line_num_abs = traceback.extract_stack()[-2].lineno
    source = textwrap.dedent(inspect.getsource(caller_frame))
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and hasattr(node.func, 'id') and node.func.id == func_name and node.lineno + caller_start_line_num - 1 == call_line_num_abs:
            return {arg.id: val for arg, val in zip(node.args, args)}
