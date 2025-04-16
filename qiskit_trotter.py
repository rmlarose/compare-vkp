# 1. Split H into terms by Abelian groups: H = H_1 + ... + H_n.
# 2. Using Ethan's code, produce a diagonalizing circuit U_i for the terms in each H_i
# 3. Get the diagonalized Pauli sum \tilde{H} = U_i H_i U_i^ and exponentiate it using PauliHedral.
# 4. The Trotter circuit for each H_i is U_d exp(-i \tilde{H}_i t) U_d^.