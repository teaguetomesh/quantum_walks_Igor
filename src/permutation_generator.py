from abc import ABC, abstractmethod


class PermutationGenerator(ABC):
    """ Generates state permutations. """

    @abstractmethod
    def get_permutation(self, state: dict[str, complex]) -> dict[str, str]:
        """ Generates permutation for a given state. Returns dict of old state -> new state mapping. Permutation for the bases that have not been specified is arbitrary. """
        pass


class PermutationGeneratorDense(PermutationGenerator):
    """ Generates dense permutations. """

    def get_permutation(self, state: dict[str, complex]) -> dict[str, str]:
        """ Maps bases in the key iteration order to sequentially increasing bases. """
        num_qubits = len(next(iter(state)))
        return {basis: format(i, f'0{num_qubits}b') for i, basis in enumerate(state)}
