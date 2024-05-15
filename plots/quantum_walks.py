""" Plots for the quantum walks paper. """
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from plots.general import Line, plot_general, save_figure


def plot_cx_count_vs_num_qubits_line(method: str, num_qubits: Sequence[int], num_amplitudes: Sequence[int], color_ind: int, marker_ind: int, label: str, figure_id: int):
    data = []
    for n, m in zip(num_qubits, num_amplitudes):
        data_path = f"../data/qubits_{n}/m_{m}/cx_counts.csv"
        df = pd.read_csv(data_path)
        data.append(np.mean(df[method]))

    line = Line(num_qubits, data, color=color_ind, marker=marker_ind)
    plot_general([line], label=label, figure_id=figure_id)


def plot_cx_count_vs_num_qubits():
    methods_all = ["shp_reduced", "qiskit"]
    densities_all = [lambda n: n, lambda n: n ** 2, lambda n: 2 ** (n - 1)]
    figure_id = 0

    for method_ind, method in enumerate(methods_all):
        for density_ind, density in enumerate(densities_all):
            if method == "shp_reduced" and density_ind == 2:
                num_qubits = np.array(range(5, 10))
            else:
                num_qubits = np.array(range(5, 12))
            num_amplitudes = [densities_all[density_ind](n) for n in num_qubits]
            plot_cx_count_vs_num_qubits_line(method, num_qubits, num_amplitudes, density_ind, method_ind, "_nolabel_", figure_id)

    plt.xlabel = "n"
    plt.ylabel = "CX"
    plt.xlim(left=4.75)
    plt.ylim(bottom=10, top=10 ** 4)
    plt.yscale("log")

    circle_marker = Line2D([0], [0], linestyle="", color="k", marker="o", markersize=5, label="Quantum Walks")
    star_marker = Line2D([0], [0], linestyle="", color="k", marker="*", markersize=8, label="Qiskit")
    blue_line = Line2D([0], [0], color="b", label=r"$m = n$")
    red_line = Line2D([0], [0], color="r", label=r"$m = n^2$")
    green_line = Line2D([0], [0], color="g", label=r"$m = 2^{n-1}$")
    plt.legend(handles=[circle_marker, star_marker, blue_line, red_line, green_line])

    save_figure()


if __name__ == "__main__":
    plot_cx_count_vs_num_qubits()

    plt.show()
