#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""This module constructs Hamiltonians for the Fermi- and Bose-Hubbard models.
"""

from openfermion.ops.operators import BosonOperator, FermionOperator, QubitOperator
from openfermion.utils.indexing import down_index, up_index
from openfermion.hamiltonians.special_operators import number_operator
import numpy as np
from openfermion.transforms import jordan_wigner
#from openfermion import QubitOperator

def modified_fermi_hubbard_default_paramaters(x_dimension: int, y_dimension: int) -> FermionOperator:
    n_bands = 1
    spinless = False
    magnetic_field = 0.0
    chemical_potential = 0.0
    periodic = False

    t = 1.0         # Same-band nearest-neighbor hopping
    t_inter = 0.0   # Inter-band nearest-neighbor hopping

    U_intra = 2.0   # On-site same-band Coulomb
    U_inter = 0.0   # On-site inter-band Coulomb (can set non-zero if needed)
    V_intra = 2.0   # Off-site same-band Coulomb (dominant)
    V_inter = 0.0   # Off-site inter-band Coulomb (less than V_intra)

    tunneling = np.zeros((y_dimension, x_dimension, y_dimension, x_dimension, n_bands, n_bands), dtype=complex)
    coulomb = np.zeros((y_dimension, x_dimension, y_dimension, x_dimension, n_bands, n_bands), dtype=complex)

    def is_neighbor(x, y, x_dim, y_dim, x2, y2, periodic=False):
        dx = (x2 - x) % x_dim if periodic else (x2 - x)
        dy = (y2 - y) % y_dim if periodic else (y2 - y)
        return (abs(dx) == 1 and dy == 0) or (abs(dy) == 1 and dx == 0)

    # Fill the tunneling matrix
    for yy in range(y_dimension):
        for xx in range(x_dimension):
            for yy2 in range(y_dimension):
                for xx2 in range(x_dimension):
                    neighbor = is_neighbor(xx, yy, x_dimension, y_dimension, xx2, yy2, periodic=periodic)
                    for alpha in range(n_bands):
                        for beta in range(n_bands):
                            if xx == xx2 and yy == yy2:
                                # On-site terms
                                if alpha == beta:
                                    # On-site same-band hopping (chemical potential)
                                    tunneling[yy, xx, yy2, xx2, alpha, beta] = chemical_potential
                                else:
                                    # On-site inter-band hopping
                                    tunneling[yy, xx, yy2, xx2, alpha, beta] = 0.0
                            else:
                                # Off-site terms
                                if neighbor:
                                    if alpha == beta:
                                        # Same-band nearest-neighbor hopping
                                        tunneling[yy, xx, yy2, xx2, alpha, beta] = t
                                    else:
                                        # Inter-band nearest-neighbor hopping
                                        tunneling[yy, xx, yy2, xx2, alpha, beta] = t_inter
                                else:
                                    # No non-neighbor terms
                                    tunneling[yy, xx, yy2, xx2, alpha, beta] = 0.0

    # Fill the Coulomb matrix
    for yy in range(y_dimension):
        for xx in range(x_dimension):
            for yy2 in range(y_dimension):
                for xx2 in range(x_dimension):
                    neighbor = is_neighbor(xx, yy, x_dimension, y_dimension, xx2, yy2, periodic=periodic)
                    for alpha in range(n_bands):
                        for beta in range(n_bands):
                            if (xx == xx2 and yy == yy2):
                                # On-site terms
                                if alpha == beta:
                                    # On-site same-band Coulomb
                                    coulomb[yy, xx, yy2, xx2, alpha, beta] = U_intra
                                else:
                                    # On-site inter-band Coulomb
                                    coulomb[yy, xx, yy2, xx2, alpha, beta] = U_inter
                            else:
                                # Off-site terms
                                if neighbor:
                                    if alpha == beta:
                                        # Off-site same-band Coulomb (dominant)
                                        coulomb[yy, xx, yy2, xx2, alpha, beta] = V_intra
                                    else:
                                        # Off-site inter-band Coulomb (weaker)
                                        coulomb[yy, xx, yy2, xx2, alpha, beta] = V_inter
                                else:
                                    coulomb[yy, xx, yy2, xx2, alpha, beta] = 0.0

    # print("Tunneling matrix shape:", tunneling.shape)
    # print("Coulomb matrix shape:", coulomb.shape)

    # Construct the Hamiltonian
    return modified_fermi_hubbard(
        x_dimension, y_dimension, n_bands,
        tunneling, coulomb,
        chemical_potential=chemical_potential,
        magnetic_field=magnetic_field,
        spinless=spinless,
        periodic=periodic
    )

def modified_fermi_hubbard(x_dimension,
                           y_dimension,
                           nbands,
                           tunneling,
                           coulomb,
                           chemical_potential=0.,
                           magnetic_field=0.,
                           periodic=True,
                           spinless=False,
                           particle_hole_symmetry=False):
    r"""Return symbolic representation of a Fermi-Hubbard Hamiltonian.

    ... [rest of the docstring] ...

    """
    if nbands > 1:
        multiband = True
    else:   
        multiband = False

    if spinless:
        if multiband:
            return multi_band_modified_spinless_fermi_hubbard(x_dimension, y_dimension, nbands,
                                                              tunneling, coulomb,
                                                              chemical_potential, magnetic_field,
                                                              periodic, particle_hole_symmetry)
        else:
            return _modified_spinless_fermi_hubbard_model(x_dimension, y_dimension,
                                                          tunneling, coulomb,
                                                          chemical_potential, magnetic_field,
                                                          periodic, particle_hole_symmetry)
    else:
        return generalized_multi_band_spinful_fermi_hubbard_respack(x_dimension, y_dimension, nbands, tunneling,
                                            coulomb, chemical_potential,
                                            magnetic_field, periodic,
                                            particle_hole_symmetry)

def hubbard_holstein(x_dimension,
                     y_dimension,
                     nbands,
                     tunneling,
                     coulomb,
                     phonons,
                     electron_phonon,
                     chemical_potential=0.,
                     magnetic_field=0.,
                     periodic=True,
                     spinless=False,
                     particle_hole_symmetry=False):

    #current implementation truncates bosonic operators to oly allow up to two phonons on each site
    
    if nbands > 1:
        multiband = True
    else:   
        multiband = False

    return generalized_multi_band_spinful_fermi_hubbard_holstein_jw(
        x_dimension, 
        y_dimension, 
        nbands, 
        tunneling,
        coulomb, 
        chemical_potential,
        magnetic_field, 
        periodic,
        particle_hole_symmetry,
        phonons, 
        electron_phonon
    )



def on_site_multi_band_spinful_fermi_hubbard(x_dimension, y_dimension, n_bands, 
                                     tunneling_matrix, coulomb_matrix, 
                                     chemical_potential, magnetic_field, 
                                     periodic, particle_hole_symmetry):

    def up_index(index):
        """Return the up-orbital index given a spatial orbital index."""
        return 2 * index

    def down_index(index):
        """Return the down-orbital index given a spatial orbital index."""
        return 2 * index + 1

    # Initialize operator.
    n_sites = x_dimension * y_dimension
    n_spin_orbitals = 2 * n_sites
    hubbard_model = FermionOperator()

    # Check matrices shapes.
    # You might want to add more robust checks based on your needs.
    if not (isinstance(tunneling_matrix, (list, tuple, np.ndarray)) and 
            len(tunneling_matrix) == y_dimension and 
            all(isinstance(row, (list, tuple, np.ndarray)) and len(row) == x_dimension for row in tunneling_matrix)):
        raise ValueError("tunneling_matrix must be a matrix.")

    if not (isinstance(coulomb_matrix, (list, tuple, np.ndarray)) and 
            len(coulomb_matrix) == y_dimension and 
            all(isinstance(row, (list, tuple, np.ndarray)) and len(row) == x_dimension for row in coulomb_matrix)):
        raise ValueError("coulomb_matrix must be a matrix.")

    # Loop through sites and add terms.
    for site in range(n_sites):
        x, y = site % x_dimension, site // x_dimension
        right_neighbor = _right_neighbor(site, x_dimension, y_dimension, periodic)
        bottom_neighbor = _bottom_neighbor(site, x_dimension, y_dimension, periodic)

        # Avoid double-counting edges for periodic conditions
        if x_dimension == 2 and periodic and site % 2 == 1:
            right_neighbor = None
        if y_dimension == 2 and periodic and site >= x_dimension:
            bottom_neighbor = None

        # Hopping terms
        for alpha in range(n_bands):
            for beta in range(n_bands):
                if right_neighbor is not None:
                    hubbard_model += FermionOperator(
                        ((n_spin_orbitals * alpha + up_index(site), 1), 
                         (n_spin_orbitals * beta + up_index(right_neighbor), 0)), 
                        -tunneling_matrix[y][right_neighbor // x_dimension][alpha][beta]
                    )
                    hubbard_model += FermionOperator(
                        ((n_spin_orbitals * alpha + down_index(site), 1), 
                         (n_spin_orbitals * beta + down_index(right_neighbor), 0)), 
                        -tunneling_matrix[y][right_neighbor // x_dimension][alpha][beta]
                    )

                if bottom_neighbor is not None:
                    hubbard_model += FermionOperator(
                        ((n_spin_orbitals * alpha + up_index(site), 1), 
                         (n_spin_orbitals * beta + up_index(bottom_neighbor), 0)), 
                        -tunneling_matrix[y][bottom_neighbor // x_dimension][alpha][beta]
                    )
                    hubbard_model += FermionOperator(
                        ((n_spin_orbitals * alpha + down_index(site), 1), 
                         (n_spin_orbitals * beta + down_index(bottom_neighbor), 0)), 
                        -tunneling_matrix[y][bottom_neighbor // x_dimension][alpha][beta]
                    )

                # Chemical potential and magnetic field terms for each band
                hubbard_model += FermionOperator(
                    ((n_spin_orbitals * alpha + up_index(site), 1), 
                     (n_spin_orbitals * alpha + up_index(site), 0)), 
                    -chemical_potential - magnetic_field
                )
                hubbard_model += FermionOperator(
                    ((n_spin_orbitals * alpha + down_index(site), 1), 
                     (n_spin_orbitals * alpha + down_index(site), 0)), 
                    -chemical_potential + magnetic_field
                )

        # On-site Coulomb interactions
        for alpha in range(n_bands):
            hubbard_model += coulomb_matrix[y][y][alpha][alpha] * FermionOperator(
                ((n_spin_orbitals * alpha + up_index(site), 1),
                 (n_spin_orbitals * alpha + down_index(site), 1),
                 (n_spin_orbitals * alpha + down_index(site), 0),
                 (n_spin_orbitals * alpha + up_index(site), 0))
            )

    return hubbard_model


def generalized_multi_band_spinful_fermi_hubbard(x_dimension, y_dimension, n_bands, 
                                                tunneling_matrix, coulomb_matrix, 
                                                chemical_potential, magnetic_field, 
                                                periodic, particle_hole_symmetry):

    def up_index(index):
        """Return the up-orbital index given a spatial orbital index."""
        return 2 * index

    def down_index(index):
        """Return the down-orbital index given a spatial orbital index."""
        return 2 * index + 1

    # Initialize operator.
    n_sites = x_dimension * y_dimension
    n_spin_orbitals = 2 * n_sites
    hubbard_model = FermionOperator()

    # Loop through sites and add terms.
    for site in range(n_sites):
        x, y = site % x_dimension, site // x_dimension
        right_neighbor = _right_neighbor(site, x_dimension, y_dimension, periodic)
        bottom_neighbor = _bottom_neighbor(site, x_dimension, y_dimension, periodic)

        # Hopping terms
        for alpha in range(n_bands):
            for beta in range(n_bands):

                # Right Neighbor
                if right_neighbor is not None:
                    hubbard_model += FermionOperator(
                        ((n_spin_orbitals * alpha + up_index(site), 1), 
                         (n_spin_orbitals * beta + up_index(right_neighbor), 0)), 
                        -tunneling_matrix[y][right_neighbor // x_dimension][alpha][beta]
                    )
                    hubbard_model += FermionOperator(
                        ((n_spin_orbitals * alpha + down_index(site), 1), 
                         (n_spin_orbitals * beta + down_index(right_neighbor), 0)), 
                        -tunneling_matrix[y][right_neighbor // x_dimension][alpha][beta]
                    )

                # Bottom Neighbor
                if bottom_neighbor is not None:
                    hubbard_model += FermionOperator(
                        ((n_spin_orbitals * alpha + up_index(site), 1), 
                         (n_spin_orbitals * beta + up_index(bottom_neighbor), 0)), 
                        -tunneling_matrix[y][bottom_neighbor // x_dimension][alpha][beta]
                    )
                    hubbard_model += FermionOperator(
                        ((n_spin_orbitals * alpha + down_index(site), 1), 
                         (n_spin_orbitals * beta + down_index(bottom_neighbor), 0)), 
                        -tunneling_matrix[y][bottom_neighbor // x_dimension][alpha][beta]
                    )

                # Chemical potential and magnetic field terms for each band
                hubbard_model += FermionOperator(
                    ((n_spin_orbitals * alpha + up_index(site), 1), 
                     (n_spin_orbitals * alpha + up_index(site), 0)), 
                    -chemical_potential - magnetic_field
                )
                hubbard_model += FermionOperator(
                    ((n_spin_orbitals * alpha + down_index(site), 1), 
                     (n_spin_orbitals * alpha + down_index(site), 0)), 
                    -chemical_potential + magnetic_field
                )

        # On-site Coulomb interactions
        for alpha in range(n_bands):
            hubbard_model += coulomb_matrix[y][x][y][x][alpha][alpha] * FermionOperator(
                ((n_spin_orbitals * alpha + up_index(site), 1),
                 (n_spin_orbitals * alpha + down_index(site), 1),
                 (n_spin_orbitals * alpha + down_index(site), 0),
                 (n_spin_orbitals * alpha + up_index(site), 0))
            )

    # Inter-site Coulomb interactions
    for i in range(n_sites):
        for j in range(n_sites):
            if i != j:  # Exclude on-site terms
                xi, yi = i % x_dimension, i // x_dimension
                xj, yj = j % x_dimension, j // x_dimension
                for alpha in range(n_bands):
                    for beta in range(n_bands):
                        hubbard_model += coulomb_matrix[yi][xi][yj][xj][alpha][beta] * FermionOperator(
                            ((n_spin_orbitals * alpha + up_index(i), 1),
                             (n_spin_orbitals * beta + up_index(j), 1),
                             (n_spin_orbitals * beta + up_index(j), 0),
                             (n_spin_orbitals * alpha + up_index(i), 0))
                        )
                        hubbard_model += coulomb_matrix[yi][xi][yj][xj][alpha][beta] * FermionOperator(
                            ((n_spin_orbitals * alpha + down_index(i), 1),
                             (n_spin_orbitals * beta + down_index(j), 1),
                             (n_spin_orbitals * beta + down_index(j), 0),
                             (n_spin_orbitals * alpha + down_index(i), 0))
                        )
            
    return hubbard_model



def generalized_multi_band_spinful_fermi_hubbard_respack(x_dimension, y_dimension, n_bands, 
                                                         tunneling_matrix, coulomb_matrix, 
                                                         chemical_potential, magnetic_field, 
                                                         periodic, particle_hole_symmetry):

    # Helper functions for indexing spin orbitals.
    def up_index(index):
        """Return the up-orbital index given a spatial site index."""
        return 2 * index

    def down_index(index):
        """Return the down-orbital index given a spatial site index."""
        return 2 * index + 1

    # Initialize operator.
    from openfermion.ops import FermionOperator
    n_sites = x_dimension * y_dimension
    n_spin_orbitals = 2 * n_sites
    hubbard_model = FermionOperator()

    # Loop through sites and add terms.
    for site in range(n_sites):
        x = site % x_dimension
        y = site // x_dimension

        right_neighbor = _right_neighbor(site, x_dimension, y_dimension, periodic)
        bottom_neighbor = _bottom_neighbor(site, x_dimension, y_dimension, periodic)

        # Add hopping and on-site terms
        for alpha in range(n_bands):
            for beta in range(n_bands):

                # Right neighbor hopping terms
                if right_neighbor is not None:
                    rx = right_neighbor % x_dimension
                    ry = right_neighbor // x_dimension

                    # Up spin hopping
                    hubbard_model += FermionOperator(
                        ((n_spin_orbitals * alpha + up_index(site), 1),
                         (n_spin_orbitals * beta + up_index(right_neighbor), 0)),
                        -tunneling_matrix[y][x][ry][rx][alpha][beta]
                    )
                    hubbard_model += FermionOperator(
                        ((n_spin_orbitals * beta + up_index(right_neighbor), 1),
                         (n_spin_orbitals * alpha + up_index(site), 0)),
                        -tunneling_matrix[ry][rx][y][x][beta][alpha]
                    )

                    # Down spin hopping
                    hubbard_model += FermionOperator(
                        ((n_spin_orbitals * alpha + down_index(site), 1),
                         (n_spin_orbitals * beta + down_index(right_neighbor), 0)),
                        -tunneling_matrix[y][x][ry][rx][alpha][beta]
                    )
                    hubbard_model += FermionOperator(
                        ((n_spin_orbitals * beta + down_index(right_neighbor), 1),
                         (n_spin_orbitals * alpha + down_index(site), 0)),
                        -tunneling_matrix[ry][rx][y][x][beta][alpha]
                    )

                # Bottom neighbor hopping terms
                if bottom_neighbor is not None:
                    bx = bottom_neighbor % x_dimension
                    by = bottom_neighbor // x_dimension

                    # Up spin hopping
                    hubbard_model += FermionOperator(
                        ((n_spin_orbitals * alpha + up_index(site), 1),
                         (n_spin_orbitals * beta + up_index(bottom_neighbor), 0)),
                        -tunneling_matrix[y][x][by][bx][alpha][beta]
                    )
                    hubbard_model += FermionOperator(
                        ((n_spin_orbitals * beta + up_index(bottom_neighbor), 1),
                         (n_spin_orbitals * alpha + up_index(site), 0)),
                        -tunneling_matrix[by][bx][y][x][beta][alpha]
                    )

                    # Down spin hopping
                    hubbard_model += FermionOperator(
                        ((n_spin_orbitals * alpha + down_index(site), 1),
                         (n_spin_orbitals * beta + down_index(bottom_neighbor), 0)),
                        -tunneling_matrix[y][x][by][bx][alpha][beta]
                    )
                    hubbard_model += FermionOperator(
                        ((n_spin_orbitals * beta + down_index(bottom_neighbor), 1),
                         (n_spin_orbitals * alpha + down_index(site), 0)),
                        -tunneling_matrix[by][bx][y][x][beta][alpha]
                    )

                # On-site diagonal terms: onsite energies, chemical potential, magnetic field
                if alpha == beta:
                    onsite_energy = -tunneling_matrix[y][x][y][x][alpha][beta] - chemical_potential
                    hubbard_model += FermionOperator(
                        ((n_spin_orbitals * alpha + up_index(site), 1),
                         (n_spin_orbitals * alpha + up_index(site), 0)),
                        onsite_energy - magnetic_field
                    )
                    hubbard_model += FermionOperator(
                        ((n_spin_orbitals * alpha + down_index(site), 1),
                         (n_spin_orbitals * alpha + down_index(site), 0)),
                        onsite_energy + magnetic_field
                    )

        # On-site Coulomb interactions (same band only, can be extended if needed)
        for alpha in range(n_bands):
            U_onsite = coulomb_matrix[y][x][y][x][alpha][alpha]
            if abs(U_onsite) > 1e-14:
                up_mode = n_spin_orbitals * alpha + up_index(site)
                down_mode = n_spin_orbitals * alpha + down_index(site)
                # n_up n_down = c_up^\dagger c_down^\dagger c_down c_up
                # The normal ordering might introduce a sign, but FermionOperator handles this.
                hubbard_model += U_onsite * FermionOperator(
                    ((up_mode, 1),
                     (down_mode, 1),
                     (down_mode, 0),
                     (up_mode, 0))
                )

    # Inter-site Coulomb interactions
    # Now we add all four spin combinations: 
    # n_{iα↑} n_{jβ↑}, n_{iα↑} n_{jβ↓}, n_{iα↓} n_{jβ↑}, n_{iα↓} n_{jβ↓}
    for i in range(n_sites):
        xi, yi = i % x_dimension, i // x_dimension
        for j in range(n_sites):
            if i == j:
                continue
            xj, yj = j % x_dimension, j // x_dimension
            for alpha in range(n_bands):
                for beta in range(n_bands):
                    U_inter = coulomb_matrix[yi][xi][yj][xj][alpha][beta]
                    if abs(U_inter) > 1e-14:
                        # Up-Up
                        hubbard_model += U_inter * FermionOperator(
                            ((n_spin_orbitals * alpha + up_index(i), 1),
                             (n_spin_orbitals * beta + up_index(j), 1),
                             (n_spin_orbitals * beta + up_index(j), 0),
                             (n_spin_orbitals * alpha + up_index(i), 0))
                        )
                        # Down-Down
                        hubbard_model += U_inter * FermionOperator(
                            ((n_spin_orbitals * alpha + down_index(i), 1),
                             (n_spin_orbitals * beta + down_index(j), 1),
                             (n_spin_orbitals * beta + down_index(j), 0),
                             (n_spin_orbitals * alpha + down_index(i), 0))
                        )
                        # Up-Down
                        hubbard_model += U_inter * FermionOperator(
                            ((n_spin_orbitals * alpha + up_index(i), 1),
                             (n_spin_orbitals * beta + down_index(j), 1),
                             (n_spin_orbitals * beta + down_index(j), 0),
                             (n_spin_orbitals * alpha + up_index(i), 0))
                        )
                        # Down-Up
                        hubbard_model += U_inter * FermionOperator(
                            ((n_spin_orbitals * alpha + down_index(i), 1),
                             (n_spin_orbitals * beta + up_index(j), 1),
                             (n_spin_orbitals * beta + up_index(j), 0),
                             (n_spin_orbitals * alpha + down_index(i), 0))
                        )

    # Note: The code does not currently adjust terms for particle-hole symmetry 
    # if particle_hole_symmetry=True. You would need to shift the number operators 
    # by 1/2 for that. This is not implemented here.

    return hubbard_model



def _modified_spinless_fermi_hubbard_model(x_dimension, y_dimension, tunneling_matrix, coulomb_matrix,
                                          chemical_potential, magnetic_field, periodic,
                                          particle_hole_symmetry):

    # Initialize operator.
    n_sites = x_dimension * y_dimension
    hubbard_model = FermionOperator()

    # Convert tunneling and coulomb to arrays if they are not
    if not isinstance(tunneling_matrix, (list, tuple, np.ndarray)):
        raise ValueError("tunneling_matrix must be a matrix (list of lists or 2D numpy array).")

    if not isinstance(coulomb_matrix, (list, tuple, np.ndarray)):
        raise ValueError("coulomb_matrix must be a matrix (list of lists or 2D numpy array).")

    # Loop through sites and add terms.
    for site in range(n_sites):
        x, y = site % x_dimension, site // x_dimension

        # Get indices of right and bottom neighbors
        right_neighbor = _right_neighbor(site, x_dimension, y_dimension, periodic)
        bottom_neighbor = _bottom_neighbor(site, x_dimension, y_dimension, periodic)

        # Calculate x and y coordinates of right and bottom neighbors conditionally
        x_rn, y_rn = (right_neighbor % x_dimension, right_neighbor // x_dimension) if right_neighbor is not None else (None, None)
        x_bn, y_bn = (bottom_neighbor % x_dimension, bottom_neighbor // x_dimension) if bottom_neighbor is not None else (None, None)

        # Avoid double-counting edges when one of the dimensions is 2 and the system is periodic
        if x_dimension == 2 and periodic and site % 2 == 1:
            right_neighbor = None
        if y_dimension == 2 and periodic and site >= x_dimension:
            bottom_neighbor = None

        # Add terms that couple with neighbors to the right and bottom.
        if right_neighbor is not None:
            hubbard_model += _hopping_term(site, right_neighbor, -tunneling_matrix[y][y_rn])

            hubbard_model += _coulomb_interaction_term(n_sites, site, right_neighbor, coulomb_matrix[y][y_rn], particle_hole_symmetry)

        if bottom_neighbor is not None:
            hubbard_model += _hopping_term(site, bottom_neighbor, -tunneling_matrix[y][y_bn])

            hubbard_model += _coulomb_interaction_term(n_sites, site, bottom_neighbor, coulomb_matrix[y][y_bn], particle_hole_symmetry)

        # Add chemical potential. The magnetic field doesn't contribute.
        hubbard_model += number_operator(n_sites, site, -chemical_potential)

    return hubbard_model

def multi_band_modified_spinless_fermi_hubbard(x_dimension, y_dimension, n_bands, tunneling_matrix, coulomb_matrix,
                                               chemical_potential, magnetic_field, periodic, particle_hole_symmetry):

    # Initialize operator.
    n_sites = x_dimension * y_dimension
    hubbard_model = FermionOperator()

    # Check if the tunneling and coulomb matrices have the correct shape
    if not (isinstance(tunneling_matrix, (list, tuple, np.ndarray)) and 
            len(tunneling_matrix) == y_dimension and 
            all(isinstance(row, (list, tuple, np.ndarray)) and len(row) == x_dimension for row in tunneling_matrix)):
        raise ValueError("tunneling_matrix must be a matrix (list of lists or 2D numpy array).")

    if not (isinstance(coulomb_matrix, (list, tuple, np.ndarray)) and 
            len(coulomb_matrix) == y_dimension and 
            all(isinstance(row, (list, tuple, np.ndarray)) and len(row) == x_dimension for row in coulomb_matrix)):
        raise ValueError("coulomb_matrix must be a matrix (list of lists or 2D numpy array).")

    # Loop through sites, bands, and add terms.
    for site in range(n_sites):
        for band1 in range(n_bands):
            for band2 in range(n_bands):
                x, y = site % x_dimension, site // x_dimension

                # Get indices of right and bottom neighbors
                right_neighbor = _right_neighbor(site, x_dimension, y_dimension, periodic)
                bottom_neighbor = _bottom_neighbor(site, x_dimension, y_dimension, periodic)

                # Calculate x and y coordinates of right and bottom neighbors conditionally
                x_rn, y_rn = (right_neighbor % x_dimension, right_neighbor // x_dimension) if right_neighbor is not None else (None, None)
                x_bn, y_bn = (bottom_neighbor % x_dimension, bottom_neighbor // x_dimension) if bottom_neighbor is not None else (None, None)

                # Avoid double-counting edges when one of the dimensions is 2 and the system is periodic
                if x_dimension == 2 and periodic and site % 2 == 1:
                    right_neighbor = None
                if y_dimension == 2 and periodic and site >= x_dimension:
                    bottom_neighbor = None

                # Add terms that couple with neighbors to the right.
                if right_neighbor is not None:
                    hubbard_model += _hopping_term(site, right_neighbor, -tunneling_matrix[y][y_rn][band1][band2])
                    hubbard_model += _coulomb_interaction_term(n_sites, site, right_neighbor, coulomb_matrix[y][y_rn][band1][band2], particle_hole_symmetry)

                # Add terms that couple with neighbors to the bottom.
                if bottom_neighbor is not None:
                    hubbard_model += _hopping_term(site, bottom_neighbor, -tunneling_matrix[y][y_bn][band1][band2])
                    hubbard_model += _coulomb_interaction_term(n_sites, site, bottom_neighbor, coulomb_matrix[y][y_bn][band1][band2], particle_hole_symmetry)

                # Add chemical potential.
                hubbard_model += number_operator(n_sites, site, -chemical_potential)

    return hubbard_model

def generalized_multi_band_spinful_fermi_hubbard_holstein_jw(
        x_dimension, y_dimension, n_bands, tunneling_matrix, coulomb_matrix, 
        chemical_potential, magnetic_field, periodic, particle_hole_symmetry,
        phonon_frequency, e_ph_coupling):
    #this allows up to two phonons per site. In order to combine fermionic and bosonic operators, everything is already transformed into the
    #qubit basis, so it is not necessary to do so after calling this function
    def up_index(index):
        """Return the up-orbital index given a spatial orbital index."""
        return 2 * index

    def down_index(index):
        """Return the down-orbital index given a spatial orbital index."""
        return 2 * index + 1

    # Initialize operator.
    n_sites = x_dimension * y_dimension
    n_spin_orbitals = 2 * n_sites
    hubbard_model = FermionOperator()

    # Loop through sites and add terms.
    for site in range(n_sites):
        x, y = site % x_dimension, site // x_dimension
        right_neighbor = _right_neighbor(site, x_dimension, y_dimension, periodic)
        bottom_neighbor = _bottom_neighbor(site, x_dimension, y_dimension, periodic)

        # Hopping terms
        for alpha in range(n_bands):
            for beta in range(n_bands):

                # Right Neighbor
                if right_neighbor is not None:
                    rx, ry = right_neighbor % x_dimension, right_neighbor // x_dimension
                    hubbard_model += FermionOperator(
                        ((n_spin_orbitals * alpha + up_index(site), 1), 
                         (n_spin_orbitals * beta + up_index(right_neighbor), 0)), 
                        -tunneling_matrix[y][x][ry][rx][alpha][beta]
                    )
                    hubbard_model += FermionOperator(
                        ((n_spin_orbitals * alpha + down_index(site), 1), 
                         (n_spin_orbitals * beta + down_index(right_neighbor), 0)), 
                        -tunneling_matrix[y][x][ry][rx][alpha][beta]
                    )

                # Bottom Neighbor
                if bottom_neighbor is not None:
                    by, bx = bottom_neighbor % x_dimension, bottom_neighbor // x_dimension
                    hubbard_model += FermionOperator(
                        ((n_spin_orbitals * alpha + up_index(site), 1), 
                         (n_spin_orbitals * beta + up_index(bottom_neighbor), 0)), 
                        -tunneling_matrix[y][x][by][bx][alpha][beta]
                    )
                    hubbard_model += FermionOperator(
                        ((n_spin_orbitals * alpha + down_index(site), 1), 
                         (n_spin_orbitals * beta + down_index(bottom_neighbor), 0)), 
                        -tunneling_matrix[y][x][by][bx][alpha][beta]
                    )

                # On-site energy from tunneling_matrix diagonal and Chemical potential and magnetic field terms
                if alpha == beta:
                    hubbard_model += FermionOperator(
                        ((n_spin_orbitals * alpha + up_index(site), 1), 
                         (n_spin_orbitals * alpha + up_index(site), 0)), 
                        -tunneling_matrix[y][x][y][x][alpha][beta] - chemical_potential - magnetic_field
                    )
                    hubbard_model += FermionOperator(
                        ((n_spin_orbitals * alpha + down_index(site), 1), 
                         (n_spin_orbitals * alpha + down_index(site), 0)), 
                        -tunneling_matrix[y][x][y][x][alpha][beta] - chemical_potential + magnetic_field
                    )

        # On-site Coulomb interactions
        for alpha in range(n_bands):
            hubbard_model += coulomb_matrix[y][x][y][x][alpha][alpha] * FermionOperator(
                ((n_spin_orbitals * alpha + up_index(site), 1),
                 (n_spin_orbitals * alpha + down_index(site), 1),
                 (n_spin_orbitals * alpha + down_index(site), 0),
                 (n_spin_orbitals * alpha + up_index(site), 0))
            )

    # Inter-site Coulomb interactions
    for i in range(n_sites):
        for j in range(n_sites):
            if i != j:  # Exclude on-site terms
                xi, yi = i % x_dimension, i // x_dimension
                xj, yj = j % x_dimension, j // x_dimension
                for alpha in range(n_bands):
                    for beta in range(n_bands):
                        hubbard_model += coulomb_matrix[yi][xi][yj][xj][alpha][beta] * FermionOperator(
                            ((n_spin_orbitals * alpha + up_index(i), 1),
                             (n_spin_orbitals * beta + up_index(j), 1),
                             (n_spin_orbitals * beta + up_index(j), 0),
                             (n_spin_orbitals * alpha + up_index(i), 0))
                        )
                        hubbard_model += coulomb_matrix[yi][xi][yj][xj][alpha][beta] * FermionOperator(
                            ((n_spin_orbitals * alpha + down_index(i), 1),
                             (n_spin_orbitals * beta + down_index(j), 1),
                             (n_spin_orbitals * beta + down_index(j), 0),
                             (n_spin_orbitals * alpha + down_index(i), 0))
                        )

    # Jordan-Wigner for Hubbard model
    hubbard_qubit_model = jordan_wigner(hubbard_model)

    # Phonon term with JW transformation
    phonon_qubit_term = QubitOperator()
    for site in range(n_sites):
        phonon_qubit_term += phonon_frequency * (QubitOperator(f"Z{2*n_spin_orbitals + 2*site}") - 
                                                 QubitOperator(f"Z{2*n_spin_orbitals + 2*site + 1}") + 1)

    # Electron-phonon interaction term with JW transformation
    e_ph_qubit_term = QubitOperator()
    for site in range(n_sites):
        for alpha in range(n_bands):
            e_ph_qubit_term += e_ph_coupling * 0.5 * (
                (QubitOperator(f"Z{n_spin_orbitals * alpha + up_index(site)}") - 1) *
                (QubitOperator(f"Z{2*n_spin_orbitals + 2*site}") - 
                 QubitOperator(f"Z{2*n_spin_orbitals + 2*site + 1}") + 1)
            )
            e_ph_qubit_term += e_ph_coupling * 0.5 * (
                (QubitOperator(f"Z{n_spin_orbitals * alpha + down_index(site)}") - 1) *
                (QubitOperator(f"Z{2*n_spin_orbitals + 2*site}") - 
                 QubitOperator(f"Z{2*n_spin_orbitals + 2*site + 1}") + 1)
            )

    # Combine all terms
    combined_qubit_model = hubbard_qubit_model + phonon_qubit_term + e_ph_qubit_term

    return combined_qubit_model

def fermi_hubbard(x_dimension,
                  y_dimension,
                  tunneling,
                  coulomb,
                  chemical_potential=0.,
                  magnetic_field=0.,
                  periodic=True,
                  spinless=False,
                  particle_hole_symmetry=False):
    r"""Return symbolic representation of a Fermi-Hubbard Hamiltonian.

    The idea of this model is that some fermions move around on a grid and the
    energy of the model depends on where the fermions are.
    The Hamiltonians of this model live on a grid of dimensions
    `x_dimension` x `y_dimension`.
    The grid can have periodic boundary conditions or not.
    In the standard Fermi-Hubbard model (which we call the "spinful" model),
    there is room for an "up" fermion and a "down" fermion at each site on the
    grid. In this model, there are a total of `2N` spin-orbitals,
    where `N = x_dimension * y_dimension` is the number of sites.
    In the spinless model, there is only one spin-orbital per site
    for a total of `N`.

    The Hamiltonian for the spinful model has the form

    $$
        \begin{align}
        H = &- t \sum_{\langle i,j \rangle} \sum_{\sigma}
                     (a^\dagger_{i, \sigma} a_{j, \sigma} +
                      a^\dagger_{j, \sigma} a_{i, \sigma})
             + U \sum_{i} a^\dagger_{i, \uparrow} a_{i, \uparrow}
                         a^\dagger_{i, \downarrow} a_{i, \downarrow}
            \\
            &- \mu \sum_i \sum_{\sigma} a^\dagger_{i, \sigma} a_{i, \sigma}
             - h \sum_i (a^\dagger_{i, \uparrow} a_{i, \uparrow} -
                       a^\dagger_{i, \downarrow} a_{i, \downarrow})
        \end{align}
    $$

    where

        - The indices $\langle i, j \rangle$ run over pairs
          $i$ and $j$ of sites that are connected to each other
          in the grid
        - $\sigma \in \{\uparrow, \downarrow\}$ is the spin
        - $t$ is the tunneling amplitude
        - $U$ is the Coulomb potential
        - $\mu$ is the chemical potential
        - $h$ is the magnetic field

    One can also construct the Hamiltonian for the spinless model, which
    has the form

    $$
        H = - t \sum_{\langle i, j \rangle} (a^\dagger_i a_j + a^\dagger_j a_i)
            + U \sum_{\langle i, j \rangle} a^\dagger_i a_i a^\dagger_j a_j
            - \mu \sum_i a_i^\dagger a_i.
    $$

    Args:
        x_dimension (int): The width of the grid.
        y_dimension (int): The height of the grid.
        tunneling (float): The tunneling amplitude $t$.
        coulomb (float): The attractive local interaction strength $U$.
        chemical_potential (float, optional): The chemical potential
            $\mu$ at each site. Default value is 0.
        magnetic_field (float, optional): The magnetic field $h$
            at each site. Default value is 0. Ignored for the spinless case.
        periodic (bool, optional): If True, add periodic boundary conditions.
            Default is True.
        spinless (bool, optional): If True, return a spinless Fermi-Hubbard
            model. Default is False.
        particle_hole_symmetry (bool, optional): If False, the repulsion
            term corresponds to:

            $$
                U \sum_{k=1}^{N-1} a_k^\dagger a_k a_{k+1}^\dagger a_{k+1}
            $$

            If True, the repulsion term is replaced by:

            $$
                U \sum_{k=1}^{N-1} (a_k^\dagger a_k - \frac12)
                                   (a_{k+1}^\dagger a_{k+1} - \frac12)
            $$

            which is unchanged under a particle-hole transformation.
            Default is False

    Returns:
        hubbard_model: An instance of the FermionOperator class.
    """
    if spinless:
        return _spinless_fermi_hubbard_model(x_dimension, y_dimension,
                                             tunneling, coulomb,
                                             chemical_potential, magnetic_field,
                                             periodic, particle_hole_symmetry)
    else:
        return _spinful_fermi_hubbard_model(x_dimension, y_dimension, tunneling,
                                            coulomb, chemical_potential,
                                            magnetic_field, periodic,
                                            particle_hole_symmetry)


def _spinful_fermi_hubbard_model(x_dimension, y_dimension, tunneling, coulomb,
                                 chemical_potential, magnetic_field, periodic,
                                 particle_hole_symmetry):

    # Initialize operator.
    n_sites = x_dimension * y_dimension
    n_spin_orbitals = 2 * n_sites
    hubbard_model = FermionOperator()

    # Loop through sites and add terms.
    for site in range(n_sites):

        # Get indices of right and bottom neighbors
        right_neighbor = _right_neighbor(site, x_dimension, y_dimension,
                                         periodic)
        bottom_neighbor = _bottom_neighbor(site, x_dimension, y_dimension,
                                           periodic)

        # Avoid double-counting edges when one of the dimensions is 2
        # and the system is periodic
        if x_dimension == 2 and periodic and site % 2 == 1:
            right_neighbor = None
        if y_dimension == 2 and periodic and site >= x_dimension:
            bottom_neighbor = None

        # Add hopping terms with neighbors to the right and bottom.
        if right_neighbor is not None:
            hubbard_model += _hopping_term(up_index(site),
                                           up_index(right_neighbor), -tunneling)
            hubbard_model += _hopping_term(down_index(site),
                                           down_index(right_neighbor),
                                           -tunneling)
        if bottom_neighbor is not None:
            hubbard_model += _hopping_term(up_index(site),
                                           up_index(bottom_neighbor),
                                           -tunneling)
            hubbard_model += _hopping_term(down_index(site),
                                           down_index(bottom_neighbor),
                                           -tunneling)

        # Add local pair Coulomb interaction terms.
        hubbard_model += _coulomb_interaction_term(n_spin_orbitals,
                                                   up_index(site),
                                                   down_index(site), coulomb,
                                                   particle_hole_symmetry)

        # Add chemical potential and magnetic field terms.
        hubbard_model += number_operator(n_spin_orbitals, up_index(site),
                                         -chemical_potential - magnetic_field)
        hubbard_model += number_operator(n_spin_orbitals, down_index(site),
                                         -chemical_potential + magnetic_field)

    return hubbard_model



def _spinless_fermi_hubbard_model(x_dimension, y_dimension, tunneling, coulomb,
                                  chemical_potential, magnetic_field, periodic,
                                  particle_hole_symmetry):

    # Initialize operator.
    n_sites = x_dimension * y_dimension
    hubbard_model = FermionOperator()

    # Loop through sites and add terms.
    for site in range(n_sites):

        # Get indices of right and bottom neighbors
        right_neighbor = _right_neighbor(site, x_dimension, y_dimension,
                                         periodic)
        bottom_neighbor = _bottom_neighbor(site, x_dimension, y_dimension,
                                           periodic)

        # Avoid double-counting edges when one of the dimensions is 2
        # and the system is periodic
        if x_dimension == 2 and periodic and site % 2 == 1:
            right_neighbor = None
        if y_dimension == 2 and periodic and site >= x_dimension:
            bottom_neighbor = None

        # Add terms that couple with neighbors to the right and bottom.
        if right_neighbor is not None:
            # Add hopping term
            hubbard_model += _hopping_term(site, right_neighbor, -tunneling)
            # Add local Coulomb interaction term
            hubbard_model += _coulomb_interaction_term(n_sites, site,
                                                       right_neighbor, coulomb,
                                                       particle_hole_symmetry)
        if bottom_neighbor is not None:
            # Add hopping term
            hubbard_model += _hopping_term(site, bottom_neighbor, -tunneling)
            # Add local Coulomb interaction term
            hubbard_model += _coulomb_interaction_term(n_sites, site,
                                                       bottom_neighbor, coulomb,
                                                       particle_hole_symmetry)

        # Add chemical potential. The magnetic field doesn't contribute.
        hubbard_model += number_operator(n_sites, site, -chemical_potential)

    return hubbard_model





def bose_hubbard(x_dimension,
                 y_dimension,
                 tunneling,
                 interaction,
                 chemical_potential=0.,
                 dipole=0.,
                 periodic=True):
    r"""Return symbolic representation of a Bose-Hubbard Hamiltonian.

    In this model, bosons move around on a lattice, and the
    energy of the model depends on where the bosons are.

    The lattice is described by a 2D grid, with dimensions
    `x_dimension` x `y_dimension`. It is also possible to specify
    if the grid has periodic boundary conditions or not.

    The Hamiltonian for the Bose-Hubbard model has the form

    $$
        H = - t \sum_{\langle i, j \rangle} (b_i^\dagger b_j + b_j^\dagger b_i)
         + V \sum_{\langle i, j \rangle} b_i^\dagger b_i b_j^\dagger b_j
         + \frac{U}{2} \sum_i b_i^\dagger b_i (b_i^\dagger b_i - 1)
         - \mu \sum_i b_i^\dagger b_i.
    $$

    where

        - The indices $\langle i, j \rangle$ run over pairs
          $i$ and $j$ of nodes that are connected to each other
          in the grid
        - $t$ is the tunneling amplitude
        - $U$ is the on-site interaction potential
        - $\mu$ is the chemical potential
        - $V$ is the dipole or nearest-neighbour interaction potential

    Args:
        x_dimension (int): The width of the grid.
        y_dimension (int): The height of the grid.
        tunneling (float): The tunneling amplitude $t$.
        interaction (float): The attractive local interaction
            strength $U$.
        chemical_potential (float, optional): The chemical potential
            $\mu$ at each site. Default value is 0.
        periodic (bool, optional): If True, add periodic boundary conditions.
            Default is True.
        dipole (float): The attractive dipole interaction strength $V$.

    Returns:
        bose_hubbard_model: An instance of the BosonOperator class.
    """

    # Initialize operator.
    n_sites = x_dimension * y_dimension
    hubbard_model = BosonOperator()

    # Loop through sites and add terms.
    for site in range(n_sites):

        # Get indices of right and bottom neighbors
        right_neighbor = _right_neighbor(site, x_dimension, y_dimension,
                                         periodic)
        bottom_neighbor = _bottom_neighbor(site, x_dimension, y_dimension,
                                           periodic)

        # Avoid double-counting edges when one of the dimensions is 2
        # and the system is periodic
        if x_dimension == 2 and periodic and site % 2 == 1:
            right_neighbor = None
        if y_dimension == 2 and periodic and site >= x_dimension:
            bottom_neighbor = None

        # Add terms that couple with neighbors to the right and bottom.
        if right_neighbor is not None:
            # Add hopping term
            hubbard_model += _hopping_term(site,
                                           right_neighbor,
                                           -tunneling,
                                           bosonic=True)
            # Add local Coulomb interaction term
            hubbard_model += _coulomb_interaction_term(
                n_sites,
                site,
                right_neighbor,
                dipole,
                particle_hole_symmetry=False,
                bosonic=True)
        if bottom_neighbor is not None:
            # Add hopping term
            hubbard_model += _hopping_term(site,
                                           bottom_neighbor,
                                           -tunneling,
                                           bosonic=True)
            # Add local Coulomb interaction term
            hubbard_model += _coulomb_interaction_term(
                n_sites,
                site,
                bottom_neighbor,
                dipole,
                particle_hole_symmetry=False,
                bosonic=True)

        # Add on-site interaction.
        hubbard_model += (
            number_operator(n_sites, site, 0.5 * interaction, parity=1) *
            (number_operator(n_sites, site, parity=1) - BosonOperator(())))

        # Add chemical potential.
        hubbard_model += number_operator(n_sites,
                                         site,
                                         -chemical_potential,
                                         parity=1)

    return hubbard_model


def _hopping_term(i, j, coefficient, bosonic=False):
    op_class = BosonOperator if bosonic else FermionOperator
    hopping_term = op_class(((i, 1), (j, 0)), coefficient)
    hopping_term += op_class(((j, 1), (i, 0)), coefficient.conjugate())
    return hopping_term


def _coulomb_interaction_term(n_sites,
                              i,
                              j,
                              coefficient,
                              particle_hole_symmetry,
                              bosonic=False):
    op_class = BosonOperator if bosonic else FermionOperator
    number_operator_i = number_operator(n_sites, i, parity=2 * bosonic - 1)
    number_operator_j = number_operator(n_sites, j, parity=2 * bosonic - 1)
    if particle_hole_symmetry:
        number_operator_i -= op_class((), 0.5)
        number_operator_j -= op_class((), 0.5)
    return coefficient * number_operator_i * number_operator_j


def _right_neighbor(site, x_dimension, y_dimension, periodic):
    if x_dimension == 1:
        return None
    if (site + 1) % x_dimension == 0:
        if periodic:
            return site + 1 - x_dimension
        else:
            return None
    return site + 1


def _bottom_neighbor(site, x_dimension, y_dimension, periodic):
    if y_dimension == 1:
        return None
    if site + x_dimension + 1 > x_dimension * y_dimension:
        if periodic:
            return site + x_dimension - x_dimension * y_dimension
        else:
            return None
    return site + x_dimension
