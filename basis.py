import numpy as np
import matplotlib.pyplot as plt


class Basis:
    def __init__(self, g1, g2):

        self._g1 = g1
        self._g2 = g2

        self._overlap = [[(f1*f2).full_integration() for f1 in [g1, g2]] for f2 in [g1, g2]]

        eval, ev = np.linalg.eigh(self._overlap)

        self._coefficients = ev.T
        self._eigenvalues = eval

        self._coefficients[0, :] = self._coefficients[0, :]/np.sqrt(eval[0])
        self._coefficients[1, :] = self._coefficients[1, :]/np.sqrt(eval[1])

    def get_ao_functions(self):
        return [self._g1, self._g2]

    def get_basis_coefficients(self):
        return self._coefficients

    def _evaluate_basis_function(self, x, n):
        return np.sum([c * f._evaluate_function(x) for c, f in zip(self._coefficients[n], self.get_ao_functions())], axis=0)

    def get_basis_function(self, start, end, n, step=0.01):
        return self._evaluate_basis_function(np.arange(start, end, step), n)

    def plot_basis_function(self, start, end, n, step=0.01):
        plt.plot(np.arange(start, end, step), self.get_basis_function(start, end, n, step))

    def _get_integrate_test(self, n):
        from scipy.integrate import simps
        x = np.arange(-10, 10, 0.01)
        return simps(self.get_basis_function(-10, 10, n, 0.01) ** 2, x)

    def get_eigenvalues(self):
        return self._eigenvalues

    def get_overlap_matrix(self):
        return self._overlap

    def get_2nd_order_overlap_matrix(self):

        g1, g2 = self.get_ao_functions()
        self._2nd_overlap = [[(g1*g1*g1*g1).full_integration(), (g1*g1*g1*g2).full_integration(), (g1*g2*g1*g1).full_integration(), (g1*g2*g1*g2).full_integration()],
                             [(g1*g1*g2*g1).full_integration(), (g1*g1*g2*g2).full_integration(), (g1*g2*g2*g1).full_integration(), (g1*g2*g2*g2).full_integration()],
                             [(g2*g1*g1*g1).full_integration(), (g2*g1*g1*g2).full_integration(), (g2*g2*g1*g1).full_integration(), (g2*g2*g1*g2).full_integration()],
                             [(g2*g1*g2*g1).full_integration(), (g2*g1*g2*g2).full_integration(), (g2*g2*g2*g1).full_integration(), (g2*g2*g2*g2).full_integration()]]

        return self._2nd_overlap
