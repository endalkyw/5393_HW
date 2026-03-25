import numpy as np
import math
import random
from collections import defaultdict
import warnings
import numpy as np
import matplotlib.pyplot as plt
import json 



def rxn(rate, reactants, products, name=""):
	return {"k": float(rate), "R": dict(reactants), "P": dict(products), "name": name}

def comb(n, r):
	if n < r: return 0.0
	if r == 0: return 1.0
	if r == 1: return float(n)
	if r == 2: return float(n * (n - 1) / 2)
	num, den = 1.0, 1.0
	for i in range(r):
		num *= (n - i)
		den *= (i + 1)
	return num / den

def propensity(state, reaction):
	a = reaction["k"]
	for sp, sto in reaction["R"].items():
		n = state.get(sp, 0)
		if n < sto:
			return 0.0
		a *= comb(n, sto)
		if a <= 0.0:
			return 0.0
	return a

def gillespie_ssa(reactions, x0, tmax=1e7, max_steps=15_000_000, seed=0, stop_fn=None):
	rng = random.Random(seed)
	x = defaultdict(int, {k: int(v) for k, v in x0.items()})
	t, steps = 0.0, 0

	while t < tmax and steps < max_steps:
		if stop_fn is not None and stop_fn(x, t, steps):
			break

		props = [propensity(x, r) for r in reactions]
		a0 = sum(props)
		if a0 <= 0.0:
			break

		t += -math.log(rng.random()) / a0

		u2 = rng.random() * a0
		cum, idx = 0.0, None
		for i, a in enumerate(props):
			cum += a
			if cum >= u2:
				idx = i
				break
		if idx is None:
			break

		r = reactions[idx]
		for sp, sto in r["R"].items():
			x[sp] -= sto
		for sp, sto in r["P"].items():
			x[sp] += sto

		steps += 1

	return dict(x), t, steps



def fetch_file(file_path):
    with open(file_path, 'r') as f:
        return f.read().strip()


def parse_dot_in_file(file_path):
    content = fetch_file(file_path).strip()
    lines = [ln.strip() for ln in content.split("\n") if ln.strip()]

    ret = {}
    for idx, line in enumerate(lines):
        vals = line.split()
        if len(vals) < 3:
            warnings.warn(f"Skipping malformed line: {line}")
            continue

        name = vals[0]
        start = int(vals[1])
        cond = vals[2]

        if cond == "N":
            stop = None
        elif cond == "GE":
            if len(vals) < 4:
                raise ValueError(f"GE requires a threshold: {line}")
            stop = int(vals[3])
        else:
            warnings.warn(f"Unknown conditional type {cond} in line: {line}")
            stop = None

        ret[name] = {
            "idx": idx,
            "name": name,
            "start": start,
            "cond": cond,
            "stop": stop
        }

    return ret

def parse_dot_r_file(file_path):
    content = fetch_file(file_path).strip()
    lines = [ln.strip() for ln in content.split("\n") if ln.strip()]

    ret = {}
    for r_index, line in enumerate(lines):
        vals = line.split(":")
        if len(vals) < 3:
            warnings.warn(f"Skipping malformed reaction line: {line}")
            continue

        cons_raw = vals[0].split()
        prod_raw = vals[1].split()
        k_val = float(vals[2].strip())

        temp = {"c": [], "c_m": [], "p": [], "p_m": [], "k": k_val}

        if len(cons_raw) % 2 != 0 or len(prod_raw) % 2 != 0:
            raise ValueError(f"Stoichiometry pairs malformed in line: {line}")

        for i in range(0, len(cons_raw), 2):
            temp["c"].append(cons_raw[i])
            temp["c_m"].append(int(cons_raw[i + 1]))

        for i in range(0, len(prod_raw), 2):
            temp["p"].append(prod_raw[i])
            temp["p_m"].append(int(prod_raw[i + 1]))

        ret[r_index] = temp

    return ret


# draw the json file from problem 2 experiments 
def visualize_result(json_path):
    with open(json_path, "r") as f:
        results = json.load(f)

    # MOI range (assumes keys are "1".."10")
    moi = np.arange(1, 11)

    counts_s = np.zeros_like(moi, dtype=float)
    counts_h = np.zeros_like(moi, dtype=float)
    probs_s  = np.zeros_like(moi, dtype=float)
    probs_h  = np.zeros_like(moi, dtype=float)

    for i, m in enumerate(moi):
        v = results[str(m)]
        counts_s[i] = v["counts"]["stealth"]
        counts_h[i] = v["counts"]["hijack"]
        probs_s[i]  = v["probs"]["stealth"]
        probs_h[i]  = v["probs"]["hijack"]

    N = results[str(moi[0])].get("N", None)
    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # --- (1) Probabilities
    ax[0].plot(moi, probs_s, marker="o", linewidth=2, label="P(stealth)")
    ax[0].plot(moi, probs_h, marker="s", linewidth=2, label="P(hijack)")
    ax[0].set_ylabel("Probability")
    ax[0].set_ylim(-0.02, 1.02)
    ax[0].grid(True, alpha=0.3)
    ax[0].legend(loc="best")

    # --- (2) Counts as side-by-side bars
    w = 0.35
    ax[1].bar(moi - w/2, counts_s, width=w, label="Stealth count")
    ax[1].bar(moi + w/2, counts_h, width=w, label="Hijack count")
    ax[1].set_xlabel("MOI")
    ax[1].set_ylabel("Count")
    ax[1].set_xticks(moi)
    ax[1].grid(True, axis="y", alpha=0.3)
    ax[1].legend(loc="best")

    title = "Lambda outcome vs MOI"
    if N is not None:
        title += f" (N={int(N)} runs per MOI)"
    fig.suptitle(title)

    fig.tight_layout()
    fig.savefig("result_2.png", dpi=200)
    print("Saved: result_2.png")