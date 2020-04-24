import numpy as np
import matplotlib.pyplot as plt
from atomic_orbital import AtomicOrbital
from molecular_orbital import MolecularOrbital
from basis import Basis
from utils import get_2nd_order_overlap_matrix


# define basis functions
plt.title('Atomic orbitals')

g1 = AtomicOrbital(center=-0.3, alpha=5, pre_exponent=1)
g1.normalize_square()
g1.plot_function(-2, 2)
print('integration g1^2 {:10.5f} {:10.5f}'.format(g1._full_integration_num_square(), g1.full_integration_square()))

g2 = AtomicOrbital(center=0.4, alpha=10, pre_exponent=2)
g2.normalize_square()
g2.plot_function(-2, 2)
print('integration g2^2 {:10.5f} {:10.5f}'.format(g2._full_integration_num_square(), g2.full_integration_square()))

plt.show()

# Generate orthogonal basis from basis functions
basis = Basis(g1, g2)

plt.title('MO basis eigenfunctions')

print('Overlap_matrix')
print(np.array(basis.get_overlap_matrix()))

basis.plot_basis_function(-2, 2, 0, 0.001)
basis.plot_basis_function(-2, 2, 1, 0.001)
print('basis function 1 integral', basis._get_integrate_test(0))
print('basis function 2 integral', basis._get_integrate_test(1))

plt.show()

# Create MO defined by the orthogonal basis
mo = MolecularOrbital(mo_coefficients=[1/np.sqrt(2), 1/np.sqrt(2)], basis=basis)

plt.title('Molecular orbital')
mo.plot_function(-2, 2)
plt.show()

print('Density matrix (MO)')
print(mo.get_density_matrix_mo())
print('total density: ', np.sum(np.multiply(np.identity(2), mo.get_density_matrix_mo())))

print('Density matrix (AO)')
print(mo.get_density_matrix_ao())
print('total density: ', np.sum(np.multiply(basis.get_overlap_matrix(), mo.get_density_matrix_ao())))

print('\n------ Symmetry - MO -------')


def apply_inverse_symmetry(basis, center=0.0):
    from copy import deepcopy
    new_basis = deepcopy(basis)

    ao_list = new_basis.get_ao_functions()

    for f in ao_list:
        f._center = -1 * (f._center - center) + center

    return new_basis


new_basis = apply_inverse_symmetry(basis, center=0.2)

mo = MolecularOrbital(mo_coefficients=[1/np.sqrt(2), 1/np.sqrt(2)], basis=basis)
mo_i = MolecularOrbital(mo_coefficients=[1/np.sqrt(2), 1/np.sqrt(2)], basis=new_basis)

plt.title('MO symmetry operation applied')
mo.plot_function(-2, 2)
mo_i.plot_function(-2, 2)

plt.show()

print('overlap mo * mo: ', mo * mo)
print('overlap mo * mo_i: ', mo * mo_i)
print('CSM-like mo * mo_i: ', 100 - (mo * mo_i / (mo * mo))*100)


print('\n------- Symmetry - Electronic density -------')
# numeric integration of MO
from scipy.integrate import simps
x = np.arange(-2, 2, 0.001)
f1 = mo.get_function(-2, 2, 0.001)**2
fi = mo_i.get_function(-2, 2, 0.001) ** 2

plt.title('electronic density (MO^2)')
plt.plot(x, f1, label='MO^2')
plt.plot(x, fi, label='MO_i^2')

b1, b2 = basis.get_ao_functions()

dm1 = mo.get_density_matrix_ao()
f1_test = dm1[0, 0] * np.array(b1.get_function(-2, 2, 0.001)) ** 2 + \
          2 * dm1[1, 0] * np.multiply(b1.get_function(-2, 2, 0.001), b2.get_function(-2, 2, 0.001)) + \
          dm1[1, 1] * np.array(b2.get_function(-2, 2, 0.001)) ** 2

dm1 = mo_i.get_density_matrix_ao()
fi_test = dm1[0, 0] * np.array(b1.get_function(-2, 2, 0.001)) ** 2 + \
          2 * dm1[1, 0] * np.multiply(b1.get_function(-2, 2, 0.001), b2.get_function(-2, 2, 0.001)) + \
          dm1[1, 1] * np.array(b2.get_function(-2, 2, 0.001)) ** 2

plt.plot(x, f1_test, '--', label='MO^2 (test)')
plt.plot(x, fi_test, '--', label='MO_i^2 (test)')
plt.legend()
plt.show()

print('overlap(num) mo^2 - mo^2:', simps(np.multiply(f1, f1), x=x))
print('overlap(num) mo_i^2 - mo_i^2:', simps(np.multiply(fi, fi), x=x))
print('overlap(num) mo^2 - mo_i^2:', simps(np.multiply(f1, fi), x=x))

print(mo.get_density_matrix_mo())
print(mo_i.get_density_matrix_mo())

print('tensor product dm x dm (MO)')

tp11_mo = np.outer(mo.get_density_matrix_mo(), mo.get_density_matrix_mo())
print('mo^2 x mo^2')
print(tp11_mo)
print('trace mo x mo: ', np.trace(tp11_mo))

tpii_mo = np.outer(mo_i.get_density_matrix_mo(), mo_i.get_density_matrix_mo())
print('mo_i^2 x mo_i^2')
print(tpii_mo)
print('trace mo_i x mo_i: ', np.trace(tpii_mo))

print('\ntensor product dm x dm (AO)')
tp_11_ao = np.outer(mo.get_density_matrix_ao(), mo.get_density_matrix_ao())
tp_ii_ao = np.outer(mo_i.get_density_matrix_ao(), mo_i.get_density_matrix_ao())
tp_1i_ao = np.outer(mo.get_density_matrix_ao(), mo_i.get_density_matrix_ao())

# 2nd order overlap matrices
s_11 = np.array(get_2nd_order_overlap_matrix(basis, basis))
s_ii = np.array(get_2nd_order_overlap_matrix(new_basis, new_basis))
s_1i = np.array(get_2nd_order_overlap_matrix(basis, new_basis))

print('mo^2 x mo^2')
print(tp_11_ao)
print('overlap mo^2 - mo^2: ', np.sum(np.multiply(s_11, tp_11_ao)))
print('mo_i^2 x mo_i^2')
print(tp_ii_ao)
print('overlap mo_i^2 - mo_i^2: ', np.sum(np.multiply(s_ii, tp_ii_ao)))
print('mo^2 x mo_i^2')
print(tp_1i_ao)
print('overlap mo^2 - mo_i^2:', np.sum(np.multiply(s_1i, tp_1i_ao)))

sym_overlap = np.sum(np.multiply(s_1i, tp_1i_ao)) / np.sum(np.multiply(s_11, tp_11_ao))
print('normalized overlap density : ', sym_overlap)
print('CSM-like density: ', (1 - sym_overlap)*100)