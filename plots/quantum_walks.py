""" Plots for the quantum walks paper. """
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plots.general import Line, plot_general


def plot_cx_count_vs_num_qubits():
    methods = ["shp_reduced", "qiskit", "shp"]
    labels = ["SHP", "Qiskit", "SHP w/o CR"]
    num_qubits = np.array(list(range(3, 12)))
    num_amplitudes = num_qubits

    data = np.zeros((len(methods), len(num_qubits)))
    for n_ind, (n, m) in enumerate(zip(num_qubits, num_amplitudes)):
        data_path = f"../data/qubits_{n}/m_{m}/cx_counts.csv"
        df = pd.read_csv(data_path)
        for method_ind, method in enumerate(methods):
            data[method_ind, n_ind] = np.mean(df[method])

    lines = []
    for row_ind in range(data.shape[0]):
        line = Line(num_qubits, data[row_ind, :], color=row_ind, label=labels[row_ind])
        lines.append(line)
    plot_general(lines, ("n", "CX"))


if __name__ == "__main__":
    plot_cx_count_vs_num_qubits()

    plt.show()
