import itertools
import os
import pickle
import random
from collections import deque
from itertools import product

import numpy as np
import numpy.random as rnd
import numpy.linalg as linalg
import qiskit.quantum_info as qqi
from qiskit.quantum_info import Statevector
from numpy import ndarray


def generate_states():
    num_qubits = np.array(list(range(8, 9)))
    num_amplitudes = num_qubits ** 2
    num_states = 1000

    for n, m in zip(num_qubits, num_amplitudes):
        out_path = f"data/qubits_{n}/m_{m}/states.pkl"
        all_inds = list(range(2 ** n))
        states = []
        for i in range(num_states):
            state_vector = qqi.random_statevector(len(all_inds)).data
            zero_inds = random.sample(all_inds, len(all_inds) - m)
            state_vector[zero_inds] = 0
            state_vector /= sum(abs(amplitude) ** 2 for amplitude in state_vector) ** 0.5
            state_dict = Statevector(state_vector).to_dict()
            states.append(state_dict)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "wb") as f:
            pickle.dump(states, f)


def split_clusters(top_cluster: ndarray, num_clusters: int) -> list[ndarray]:
    all_clusters = deque([top_cluster])
    while len(all_clusters) < num_clusters:
        next_cluster = all_clusters.popleft()
        spanned_dims = np.where(next_cluster == -1)[0]
        split_dim = rnd.choice(spanned_dims)
        for split_val in [0, 1]:
            subcluster = next_cluster.copy()
            subcluster[split_dim] = split_val
            all_clusters.append(subcluster)
    return list(all_clusters)


def clusters_overlap(cluster1: ndarray, cluster2: ndarray) -> bool:
    overlap = True
    for c1, c2 in zip(cluster1, cluster2):
        if c1 != -1 and c2 != -1 and c1 != c2:
            overlap = False
            break
    return overlap


def move_clusters(all_clusters: list[ndarray]):
    for i in range(len(all_clusters)):
        cluster = all_clusters[i]
        fixed_dims = cluster != -1
        num_fixed = np.sum(fixed_dims)
        position_found = False
        while not position_found:
            cluster[fixed_dims] = rnd.randint(0, 2, num_fixed)
            for j in range(i):
                if clusters_overlap(cluster, all_clusters[j]):
                    break
            else:
                position_found = True


def get_all_cluster_states(cluster: ndarray) -> list[str]:
    num_spanned_dims = np.sum(cluster == -1)
    state_labels = []
    for i in range(2 ** num_spanned_dims):
        i_bin = [int(c) for c in format(i, f'0{num_spanned_dims}b')]
        state_label = cluster.copy()
        state_label[state_label == -1] = i_bin
        state_labels.append(''.join([str(val) for val in state_label]))
    return state_labels


def generate_state(all_clusters: list[ndarray]) -> dict[str, complex]:
    all_state_labels = list(itertools.chain.from_iterable([get_all_cluster_states(cluster) for cluster in all_clusters]))
    all_amplitudes = rnd.uniform(0, 1, len(all_state_labels)) * np.exp(-1j * rnd.uniform(0, 2 * np.pi, len(all_state_labels)))
    all_amplitudes /= linalg.norm(all_amplitudes)
    state = {label: amplitude for label, amplitude in zip(all_state_labels, all_amplitudes)}
    return state


def generate_cluster_states_consistent():
    """ Generates cluster states where cluster dimensions are random, but consistent, i.e. assemblable into a dense hypercube without rotation. """
    num_states = 100
    num_qubits = 10
    num_qubits_dense = 5
    num_clusters = 32
    out_path = f'data_2/qubits_{num_qubits}/dense_{num_qubits_dense}/clusters_{num_clusters}/states.pkl'

    states = []
    for i in range(num_states):
        top_cluster = np.zeros(num_qubits, dtype=int)
        top_cluster[random.sample(list(range(num_qubits)), num_qubits_dense)] = -1
        all_clusters = split_clusters(top_cluster, num_clusters)
        move_clusters(all_clusters)
        states.append(generate_state(all_clusters))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(states, f)


def generate_cluster_states_same():
    """ Generates cluster states where cluster dimensions are the same for all clusters of any state. """
    num_states = 100
    num_qubits_dense_all = [7]
    num_qubits_all = [12]
    num_clusters_all = [2, 4, 8, 16, 32, 64, 128]

    generator = np.random.default_rng()
    iterable = list(product(num_qubits_all, num_qubits_dense_all, num_clusters_all))
    for num_qubits, num_qubits_dense, num_clusters in iterable:
        print(f'Qubits: {num_qubits}, Dense: {num_qubits_dense}, Clusters: {num_clusters}')
        out_path = f'data_3/qubits_{num_qubits}/dense_{num_qubits_dense}/clusters_{num_clusters}/states.pkl'
        states = []
        for i in range(num_states):
            num_cluster_dims = int(np.log2(2 ** num_qubits_dense / num_clusters))
            cluster_dims = generator.choice(num_qubits, num_cluster_dims, replace=False, shuffle=False)
            clusters = []
            for j in range(num_clusters):
                while True:
                    next_cluster = generator.choice(2, num_qubits)
                    next_cluster[cluster_dims] = -1
                    if not any([np.all(cluster == next_cluster) for cluster in clusters]):
                        break
                clusters.append(next_cluster)
            states.append(generate_state(clusters))

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "wb") as f:
            pickle.dump(states, f)


def generate_cluster_states_random():
    """ Generates cluster states where cluster dimensions are chosen randomly. """
    num_states = 100
    num_qubits_all = [10]
    num_qubits_dense_all = [5]
    num_clusters_all = [2, 4, 8, 16, 32]

    generator = np.random.default_rng()
    iterable = list(product(num_qubits_all, num_qubits_dense_all, num_clusters_all))
    for num_qubits, num_qubits_dense, num_clusters in iterable:
        print(f'Qubits: {num_qubits}, Dense: {num_qubits_dense}, Clusters: {num_clusters}')
        out_path = f'data_4/qubits_{num_qubits}/dense_{num_qubits_dense}/clusters_{num_clusters}/states.pkl'
        states = []
        for i in range(num_states):
            num_cluster_dims = int(np.log2(2 ** num_qubits_dense / num_clusters))
            clusters = []
            for j in range(num_clusters):
                while True:
                    cluster_dims = generator.choice(num_qubits, num_cluster_dims, replace=False, shuffle=False)
                    next_cluster = generator.choice(2, num_qubits)
                    next_cluster[cluster_dims] = -1
                    if not any([clusters_overlap(cluster, next_cluster) for cluster in clusters]):
                        break
                clusters.append(next_cluster)
            states.append(generate_state(clusters))

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "wb") as f:
            pickle.dump(states, f)


if __name__ == "__main__":
    generate_cluster_states_same()
