""" Plots for the permutation unitaries paper. """
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from plots.general import Line, plot_general, save_figure


def plot_cx_count_vs_clusters():
    methods = ['qiskit_dense', 'shp', 'qiskit']
    labels = ['Qiskit Dense', 'Single Edge', 'Qiskit Default']
    num_qubits = [8, 9, 10]
    num_qubits_dense = 5
    num_clusters_all = [2, 4, 8, 16, 32]

    lines = []
    for num_qubits_ind, num_qubits in enumerate(num_qubits):
        for method_ind, method in enumerate(methods):
            cx_counts = []
            for num_clusters in num_clusters_all:
                data_path = f'../data_2/qubits_{num_qubits}/dense_{num_qubits_dense}/clusters_{num_clusters}/cx_counts.csv'
                df = pd.read_csv(data_path)
                cx_counts.append(np.mean(df[method]))
            label = labels[method_ind] if num_qubits_ind == 0 else None
            lines.append(Line(num_clusters_all, cx_counts, color=method_ind, style=num_qubits_ind, marker=0, label=label))
    plot_general(lines, ("Clusters", "CX"), boundaries=None)
    save_figure()


if __name__ == "__main__":
    plot_cx_count_vs_clusters()
    plt.show()
