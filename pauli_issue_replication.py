from typing import List
import pickle
import cirq

qs = cirq.LineQubit.range(3)
ps: List[cirq.PauliString] = []
ps.append(1.0 * cirq.Z(qs[0]) * cirq.Y(qs[1]))
ps.append(-0.5 * cirq.X(qs[1]) * cirq.Y(qs[2]))

with open("example_groups.pkl", "wb") as f:
    pickle.dump(ps, f)

with open("example_groups.pkl", "rb") as f:
    new_ps = pickle.load(f)

print(new_ps[0] * new_ps[1])