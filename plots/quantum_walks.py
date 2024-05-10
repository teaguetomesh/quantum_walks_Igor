""" Plots for the quantum walks paper. """
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plots.general import Line, plot_general, save_figure


def plot_cx_count_vs_num_qubits_line(method: str, num_qubits: Sequence[int], num_amplitudes: Sequence[int], color_ind: int, marker_ind: int, label: str, figure_id: int):
    data = []
    for n, m in zip(num_qubits, num_amplitudes):
        data_path = f"../data/qubits_{n}/m_{m}/cx_counts.csv"
        df = pd.read_csv(data_path)
        data.append(np.mean(df[method]))

    line = Line(num_qubits, data, color=color_ind, marker=marker_ind, label=label)
    plot_general([line], ("n", "CX"), boundaries=(None, None, None, 5000), figure_id=figure_id)


def plot_cx_count_vs_num_qubits():
    methods_all = [["shp_reduced", "qiskit", "shp"], ["shp_reduced", "qiskit"], ["shp_reduced", "qiskit"]]
    labels_all = [["SHP", "Qiskit", "SHP w/o CR"], None, None]
    num_qubits_all = [np.array(range(3, 12)), np.array(range(5, 12)), np.array(range(3, 9))]
    num_amplitudes_all = [num_qubits_all[0], num_qubits_all[1] ** 2, 2 ** (num_qubits_all[2] - 1)]
    figure_id = 0

    for density_ind in range(len(num_qubits_all)):
        methods = methods_all[density_ind]
        labels = labels_all[density_ind]
        num_qubits = num_qubits_all[density_ind]
        num_amplitudes = num_amplitudes_all[density_ind]
        for method_ind, method in enumerate(methods):
            label = labels[method_ind] if labels is not None else "_nolabel_"
            plot_cx_count_vs_num_qubits_line(method, num_qubits, num_amplitudes, method_ind, density_ind, label, figure_id)
    save_figure()


if __name__ == "__main__":
    plot_cx_count_vs_num_qubits()

    plt.show()
