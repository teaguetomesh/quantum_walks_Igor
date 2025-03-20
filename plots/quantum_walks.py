""" Plots for the quantum walks paper. """
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from general import Line, plot_general, save_figure
from src.utilities.general import get_error_margin


def plot_cx_count_vs_num_qubits_line(method: str, num_qubits: Sequence[int], num_amplitudes: Sequence[int], color_ind: int, marker_ind: int, label: str, figure_id: int):
    ys = []
    error_margins = []
    for n, m in zip(num_qubits, num_amplitudes):
        data_path = f"../data/qubits_{n}/m_{m}/cx_counts.csv"
        df = pd.read_csv(data_path)
        ys.append(np.mean(df[method]))
        error_margins.append(get_error_margin(df[method]))

    line = Line(num_qubits, ys, error_margins=error_margins, color=color_ind, marker=marker_ind, label=label)
    plot_general([line], ("n", "CX"), boundaries=(4.75, 11.25, 10, 10 ** 4), figure_id=figure_id)
    plt.yscale("log")


def plot_control_reduction_effect():
    num_qubits = np.array(range(5, 12))
    num_amplitudes = num_qubits
    figure_id = 0
    plot_cx_count_vs_num_qubits_line("random_reduced", num_qubits, num_amplitudes, 0, 0, "With control reduction", figure_id)
    plot_cx_count_vs_num_qubits_line("random", num_qubits, num_amplitudes, 1, 0, "Without control reduction", figure_id)
    save_figure()


def plot_walk_order_comparison():
    num_qubits = np.array(range(5, 12))
    num_amplitudes = num_qubits
    figure_id = 0
    methods = ["random", "mst", "shp", "linear"]
    methods = [method + "_reduced" for method in methods]
    labels = ["Random", "MST", "SHP", "Sorted"]
    for method_ind, method in enumerate(methods):
        plot_cx_count_vs_num_qubits_line(method, num_qubits, num_amplitudes, method_ind, 0, labels[method_ind], figure_id)
    # plt.yscale("linear")
    plt.ylim(top=175)
    save_figure()


def plot_qiskit_comparison():
    methods_all = ["shp_reduced", "qiskit", "qiskit_dense", "multiedge"]
    densities_all = [lambda n: n, lambda n: n ** 2, lambda n: 2 ** (n - 1)]
    figure_id = 0

    for method_ind, method in enumerate(methods_all):
        density_inds = [0, 1]
        for density_ind in density_inds:
            num_qubits = np.array(range(5, 12))
            num_amplitudes = [densities_all[density_ind](n) for n in num_qubits]
            plot_cx_count_vs_num_qubits_line(method, num_qubits, num_amplitudes, density_ind, method_ind, "_nolabel_", figure_id)

    circle_marker = Line2D([0], [0], linestyle="", color="k", marker="o", markersize=10, label="Single-Edge")
    star_marker = Line2D([0], [0], linestyle="", color="k", marker="*", markersize=10, label="Qiskit")
    x_marker = Line2D([0], [0], linestyle="", color="k", marker="X", markersize=10, label="Qiskit Dense")
    triangle_marker = Line2D([0], [0], linestyle="", color="k", marker="v", markersize=10, label="Multi-Edge")
    blue_line = Line2D([0], [0], color="b", label=r"$m = n$")
    red_line = Line2D([0], [0], color="r", label=r"$m = n^2$")
    green_line = Line2D([0], [0], color="g", label=r"$m = 2^{n-1}$")
    plt.legend(handles=[circle_marker, star_marker, x_marker, triangle_marker, blue_line, red_line], draggable=True)
    save_figure()


if __name__ == "__main__":
    # plot_control_reduction_effect()
    # plot_walk_order_comparison()
    plot_qiskit_comparison()

    plt.show()
