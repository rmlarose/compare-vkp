import argparse
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("l", type=int, help="Lattice dimension.")
    parser.add_argument("u_t_ratio", type=float, help="Ratio U/t.")
    args = parser.parse_args()

    steps = [10, 20, 40, 50, 60, 100, 150, 200, 250, 300, 350, 400]

    l = args.l
    n_occ = (l * l) / 2 # Half filling
    t = 1.0
    u = args.u_t_ratio
    alpha = 3.0
    max_bond = 100
    tau = 0.1
    d = 16
    ratio = 0.5
    eps = 1e-8

    for i, s in enumerate(steps):
        param_dict = {
            "l": l,
            "n_occ": n_occ,
            "t": t,
            "u": u,
            "alpha": alpha,
            "max_bond": max_bond,
            "tau": tau,
            "d": d,
            "ratio": ratio,
            "eps": eps,
            "steps": s
        }
        with open(f"input_{i}.json", "w", encoding="UTF8") as f:
            json.dump(param_dict, f)

if __name__ == "__main__":
    main()