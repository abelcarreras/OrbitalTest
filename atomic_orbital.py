import numpy as np
import matplotlib.pyplot as plt


class AtomicOrbital:
    def __init__(self, center, alpha, pre_exponent=1):
        self._center = center
        self._alpha = alpha
        self._pre_exponent = pre_exponent

    def _evaluate_function(self, x):
        return self._pre_exponent * np.exp(-self._alpha*(x - self._center)**2)

    def get_function(self, start, end, step=0.01):
        return self._evaluate_function(np.arange(start, end, step))

    def plot_function(self, start, end, step=0.01):
        plt.plot(np.arange(start, end, step), self.get_function(start, end, step))
        # plt.show()

    def full_integration(self):
        return self._pre_exponent * np.sqrt(np.pi/self._alpha)

    def full_integration_square(self):
        return self._pre_exponent * np.sqrt(np.pi/self._alpha**0.5/np.sqrt(np.pi*2))

    def _full_integration_num_square(self):
        # Numerical integral for testing purposes
        from scipy.integrate import simps
        x = np.arange(-10, 10, 0.01)
        return simps(self.get_function(-10, 10, 0.01)**2, x=x)

    def _full_integration_num(self):
        # Numerical integral for testing purposes
        from scipy.integrate import simps
        x = np.arange(-10, 10, 0.01)
        return simps(self.get_function(-10, 10, 0.01), x=x)

    def normalize_square(self):
        self._pre_exponent /= self.full_integration_square()

    def __mul__(self, other):
        # define gaussian product
        if type(other) == AtomicOrbital:
            # define gaussian product
            dist = self._center - other._center
            new_alpha = self._alpha + other._alpha
            new_center = (self._alpha * self._center + other._alpha * other._center) / new_alpha
            new_pre_exponent = np.exp(-self._alpha * other._alpha * dist**2/new_alpha) * self._pre_exponent * other._pre_exponent

            return AtomicOrbital(new_center, new_alpha, new_pre_exponent)

        else:
            return AtomicOrbital(self._center, self._alpha, self._pre_exponent * other)
