import numpy as np
import matplotlib.pyplot as plt
from atomic_orbital import AtomicOrbital
from molecular_orbital import MolecularOrbital
from basis import Basis
from utils import get_2nd_order_overlap_matrix


# define basis functions
plt.title('Atomic orbitals')

g1 = AtomicOrbital(center=-0.4, alpha=5, pre_exponent=1)
g1.normalize_square()
g1.plot_function(-2, 2)
print('integration g1', g1._full_integration_num_square(), g1.full_integration_square())

g2 = AtomicOrbital(center=0.4, alpha=10, pre_exponent=2)
g2.normalize_square()
g2.plot_function(-2, 2)
print('integration g2', g2._full_integration_num_square(), g2.full_integration_square())

plt.show()

# Generate orthogonal basis from basis functions
basis = Basis(g1, g2)

plt.title('Basis functions')

print(basis.get_eigenvalues())
print('Overlap_matrix')
print(np.array(basis.get_overlap_matrix()))

basis.plot_basis_function(-5, 5, 0, 0.001)
basis.plot_basis_function(-5, 5, 1, 0.001)
print('basis1', basis._get_integrate_test(0))
print('basis2', basis._get_integrate_test(1))

plt.show()

# Create MO defined by the orthogonal basis
mo = MolecularOrbital(mo_coefficients=[1/np.sqrt(2), 1/np.sqrt(2)], basis=basis)
mo2 = MolecularOrbital(mo_coefficients=[1/np.sqrt(2), -1/np.sqrt(2)], basis=basis)

plt.title('Molecular orbitals')
mo.plot_function(-2, 2)
mo2.plot_function(-2, 2)
plt.show()

print('test_square: ', mo._full_integration_num_square())
print('dot 1*1:', mo * mo)
print('dot 1*2:', mo * mo2)
print('dot 2*2:', mo2 * mo2)
print('dot 1*2', np.dot(mo.get_ao_coefficients(), np.dot(basis.get_overlap_matrix(), mo2.get_ao_coefficients())))

plt.title('Molecular orbitals')
overlap = []
for a in np.arange(0, 2*np.pi, 0.5):
    mo2 = MolecularOrbital(mo_coefficients=[np.sin(a), np.cos(a)], basis=basis)
    overlap.append(mo * mo2)
    mo2.plot_function(-2, 2, 0.01)
plt.show()

plt.title('MO overlap')
plt.plot(np.arange(0, 2*np.pi, 0.5), overlap)
plt.show()

print('integral m1', mo._full_integration_num(), mo.full_integration())
print('integral m2', mo2._full_integration_num(), mo2.full_integration())

print('Density matrix (MO)')
print(mo.get_density_matrix_mo())
print(np.sum(np.multiply(np.identity(2), mo.get_density_matrix_mo())))

print('Density matrix (AO)')
print(mo2.get_density_matrix_ao())

print(np.sum(np.multiply(basis.get_overlap_matrix(), mo.get_density_matrix_ao())))


def apply_inverse_symmetry(basis, center=0.0):
    from copy import deepcopy
    new_basis = deepcopy(basis)

    ao_list = new_basis.get_ao_functions()

    for f in ao_list:
        f._center = -1 * (f._center - center) + center

    return new_basis


print('\n-------symmetry (MO) -------')
new_basis = apply_inverse_symmetry(basis, center=0.2)

mo = MolecularOrbital(mo_coefficients=[1/np.sqrt(2), 1/np.sqrt(2)], basis=basis)
mo2 = MolecularOrbital(mo_coefficients=[1/np.sqrt(2), -1/np.sqrt(2)], basis=basis)

mo_i = MolecularOrbital(mo_coefficients=[1/np.sqrt(2), 1/np.sqrt(2)], basis=new_basis)


plt.title('Molecular orbitals (symmetry)')
mo.plot_function(-2, 2)
mo_i.plot_function(-2, 2)
#mo2.plot_function(-2, 2)

plt.show()

print('overlap mo * mo: ', mo * mo)
print('overlap mo * mo2: ', mo * mo2)
print('overlap mo * mo_i: ', mo * mo_i)
print('CSM-like mo * mo_i: ', 100 - (mo * mo_i / (mo * mo))*100)


from scipy.integrate import simps
x = np.arange(-5, 5, 0.001)
f1 = mo.get_function(-5, 5, 0.001)**2
f2 = mo2.get_function(-5, 5, 0.001)**2
f1_i = mo_i.get_function(-5, 5, 0.001)**2

plt.plot(x, f1)
#plt.plot(x, f2)
#plt.plot(x, f1_i)

b1, b2 = basis.get_ao_functions()

dm1 = mo.get_density_matrix_ao()
f1_test = dm1[0, 0] * np.array(b1.get_function(-5, 5, 0.001)) ** 2 + \
          2 * dm1[1, 0] * np.multiply(b1.get_function(-5, 5, 0.001), b2.get_function(-5, 5, 0.001)) + \
          dm1[1, 1] * np.array(b2.get_function(-5, 5, 0.001)) ** 2

plt.plot(x, f1_test, '--')

plt.show()

print('\n------- symmetry (Electronic density) -------')

print('overlap(num) mo^2 - mo_i^2:', simps(np.multiply(f1, f1_i), x=x))
print('overlap(num) mo^2 - mo2^2:', simps(np.multiply(f1, f2), x=x))

dm1 = mo.get_density_matrix_ao()
dm2 = mo2.get_density_matrix_ao()

print('AO dm contraction (mo): ', np.sum(np.multiply(basis.get_overlap_matrix(), dm1)))
print('AO dm contraction (mo2): ', np.sum(np.multiply(basis.get_overlap_matrix(), dm2)))

print(mo.get_density_matrix_mo())
print(mo2.get_density_matrix_mo())

print('--------------------------')

print('tensor product dm x dm (MO)')

tp11_mo = np.outer(mo.get_density_matrix_mo(), mo.get_density_matrix_mo())
print('mo x mo')
print(tp11_mo)
print('trace mo x mo: ', np.trace(tp11_mo))
tp12_mo = np.outer(mo.get_density_matrix_mo(), mo2.get_density_matrix_mo())
print('mo x mo2')
print(tp12_mo)
print('trace mo x mo2: ', np.trace(tp12_mo))

print('\ntensor product pdm1 x pdm1 (AO)')
tp_11_ao = np.outer(mo.get_density_matrix_ao(), mo.get_density_matrix_ao())
tp_22_ao = np.outer(mo2.get_density_matrix_ao(), mo2.get_density_matrix_ao())

tp_12ao = np.outer(mo.get_density_matrix_ao(), mo2.get_density_matrix_ao())
tp_11i_ao = np.outer(mo.get_density_matrix_ao(), mo_i.get_density_matrix_ao())



s2 = np.array(get_2nd_order_overlap_matrix(basis, basis))
s2b = np.array(get_2nd_order_overlap_matrix(basis, new_basis))

print('mo x mo')
print(tp_11_ao)
print('overlap 1^2 - 1^2: ', np.sum(np.multiply(s2, tp_11_ao)))
print('mo x mo2')
print(tp_12ao)
print('overlap 1^2 - 2^2: ', np.sum(np.multiply(s2, tp_12ao)))
print('mo x mo_i')
print(tp_11i_ao)
print('overlap 1^2 - 1i^2:', np.sum(np.multiply(s2b, tp_11i_ao)))

sym_overlap = np.sum(np.multiply(s2b, tp_11i_ao)) / np.sum(np.multiply(s2, tp_11_ao))
print('symmetry_overlap (inverse): ', sym_overlap)
print('CSM-like (inverse): ', (1 - sym_overlap)*100)