""" General utilities. """
import ast
import inspect
import textwrap
import traceback
from dataclasses import dataclass

import numpy as np
from numpy import ndarray
from numpy.random.mtrand import Sequence


@dataclass
class GreedyNode:
    cost: int | float
    input: list[int] | None
    output: tuple | None


def greedy_decision_tree(target_func: callable, allowed_vals: list[int], stop_no_improvement: bool = False, max_vals: int = None) -> list[GreedyNode]:
    """ Takes a function that takes list of integers with allowed values from allowed_vals and returns a tuple with cost as the 0th element.
    Tries to greedily find the input that minimizes cost by adding 1 number at a time to the list. Each value from allowed_vals can only be added once.
    If stop_no_improvement is True, stops exploring the tree if current level did not find a better solution compared to the previous level.
    If max_vals is not None, stops exploring the tree after reaching the level with max_vals values in the input. Otherwise, continues until all allowed values are exhausted.
    Returns a list of the best nodes at each level of the greedy decision tree. """
    best_sequence = [GreedyNode(np.inf, [], None)]
    remaining_vals = set(allowed_vals)
    while len(remaining_vals) > 0:
        best_this_level = GreedyNode(np.inf, None, None)
        for val in remaining_vals:
            input = best_sequence[-1].input + [val]
            output = target_func(input)
            cost = output[0]
            if cost < best_this_level.cost:
                best_this_level = GreedyNode(cost, input, output[1:])
        if stop_no_improvement and best_this_level.cost >= best_sequence[-1].cost:
            break
        best_sequence.append(best_this_level)
        if max_vals is not None and len(best_this_level.input) == max_vals:
            break
        remaining_vals.remove(best_this_level.input[-1])
    return best_sequence[1:]


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
