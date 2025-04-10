""" Plots for the permutation unitaries paper. """
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from plots.general import Line, plot_general, save_figure
from src.utilities.general import get_error_margin


def plot_cx_count_vs_clusters():
    methods = ['qiskit_dense', 'pairwise_swap', 'merging_states', 'qiskit']
    labels = ['Cluster Swaps', 'Pairwise Swaps', 'Merging States', 'Qiskit']
    num_qubits_all = [10, 11, 12]
    num_qubits_dense_all = [5, 6, 7]
    num_clusters_all = [2, 4, 8, 16, 32, 64, 128]

    lines = []
    for num_qubits_ind, num_qubits in enumerate(num_qubits_all):
        num_qubits_dense = num_qubits_dense_all[num_qubits_ind]
        num_clusters_next = num_clusters_all[:num_qubits_dense]
        for method_ind, method in enumerate(methods):
            average_degrees = []
            cx_counts = []
            error_margins = []
            for num_clusters in num_clusters_next:
                data_path = f'../data_3/qubits_{num_qubits}/dense_{num_qubits_dense}/clusters_{num_clusters}/cx_counts.csv'
                df = pd.read_csv(data_path)
                average_degrees.append(np.mean(df['average_degree']))
                cx_counts.append(np.mean(df[method]))
                error_margins.append(get_error_margin(df[method]))
            label = labels[method_ind] if num_qubits_ind == 0 else None
            lines.append(Line(average_degrees, cx_counts, error_margins, color=method_ind, marker=num_qubits_ind, label=label))
    plot_general(lines, ('Average neighbors', 'Average CX'), boundaries=None)
    save_figure()


if __name__ == "__main__":
    plot_cx_count_vs_clusters()
    plt.show()
