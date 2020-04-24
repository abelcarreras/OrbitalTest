import numpy as np
import matplotlib.pyplot as plt


class MolecularOrbital:
    def __init__(self, mo_coefficients, basis):
        self._basis = basis
        self._mo_coefficients = mo_coefficients
        basis_coefficients = self._basis.get_basis_coefficients()
        self._ao_coefficients = np.dot(self._mo_coefficients, basis_coefficients)

        np.testing.assert_almost_equal(np.dot(mo_coefficients, mo_coefficients), 1.0)

    def get_mo_coefficients(self):
        return self._mo_coefficients

    def get_ao_coefficients(self):
        return self._ao_coefficients

    def _evaluate_function_old(self, x):
        return np.sum([c * f._evaluate_function(x)
                       for c, f in zip(self._ao_coefficients, self._basis.get_ao_functions())], axis=0)

    def _get_function_old(self, start, end, step=0.01):
        return self._evaluate_function_old(np.arange(start, end, step))

    def get_function(self, start, end, step=0.01):
        return np.sum([self._mo_coefficients[i] * np.array(self._basis.get_basis_function(start, end, i, step=step))
                       for i in range(2)], axis=0)

    def plot_function(self, start, end, step=0.01):
        plt.plot(np.arange(start, end, step), self.get_function(start, end, step))

    def full_integration(self):
        return np.sum([c * bf.full_integration()
                       for c, bf in zip(self._ao_coefficients, self._basis.get_ao_functions())])

    def _full_integration_num(self):
        # Numerical integral for testing purposes
        from scipy.integrate import simps
        x = np.arange(-10, 10, 0.01)
        return simps(self.get_function(-10, 10, 0.01), x=x)

    def _full_integration_num_square(self):
        # Numerical integral for testing purposes
        from scipy.integrate import simps
        x = np.arange(-10, 10, 0.01)
        return simps(self.get_function(-10, 10, 0.01)**2, x=x)

    def get_density_matrix_mo(self):
        return np.outer(self._mo_coefficients, self._mo_coefficients)

    def get_density_matrix_ao(self):
        return np.outer(self._ao_coefficients, self._ao_coefficients)

    def _get_density_matrix_ao_alternative(self):
        return np.dot(self._basis.get_basis_coefficients().T,
                      np.dot(self.get_density_matrix_mo(),
                             self._basis.get_basis_coefficients()))

    def __mul__(self, other):

        if type(other) == MolecularOrbital:
            # define scalar product between two MO

            option = 1
            if option == 0:
                # using calculus
                from scipy.integrate import simps

                x = np.arange(-10, 10, 0.01)
                f1 = self.get_function(-10, 10, 0.01)
                f2 = other.get_basis_function(-10, 10, 0.01)

                return simps(np.multiply(f1, f2), x)

            elif option == 1:
                # using linear algebra
                ao_coeff_1 = self._ao_coefficients
                ao_coeff_2 = other._ao_coefficients

                bf_1 = self._basis.get_ao_functions()
                bf_2 = other._basis.get_ao_functions()

                return np.sum([ao_coeff_1[0] * ao_coeff_2[0] * (bf_1[0] * bf_2[0]).full_integration(),
                               ao_coeff_1[0] * ao_coeff_2[1] * (bf_1[0] * bf_2[1]).full_integration(),
                               ao_coeff_1[1] * ao_coeff_2[0] * (bf_1[1] * bf_2[0]).full_integration(),
                               ao_coeff_1[1] * ao_coeff_2[1] * (bf_1[1] * bf_2[1]).full_integration()])

            elif option == 2:
                # straightforward
                return np.dot(other._mo_coefficients, self._mo_coefficients)

            elif option == 3:
                # using gaussian (AO) basis
                s = self._basis.get_overlap_matrix()
                return np.dot(other._ao_coefficients, np.dot(s, self._ao_coefficients))
