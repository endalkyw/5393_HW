import math
import random
from collections import defaultdict
import pandas as pd
from helper import rxn, comb, propensity, gillespie_ssa


# log and exp modules -------------------------

def build_log_on_y_guarded(slow_clock=3e-4, faster=3e5, fast=3e3, medium=10.0):
	result = [
		# Guarded serialized clock: requires 2y but returns it unchanged
		rxn(slow_clock, {"S": 1, "b": 1, "y": 2}, {"a": 1, "b": 1, "y": 2}, "S+b+2y -> a+b+2y"),

		# Core halving + counter
		rxn(faster, {"a": 1, "y": 2}, {"c": 1, "yp": 1, "a": 1}, "a+2y -> c+yp+a"),
		rxn(faster, {"c": 2}, {"c": 1}, "2c -> c"),

		# Cleanup + promotion
		rxn(fast,   {"a": 1}, {}, "a -> 0"),
		rxn(medium, {"yp": 1}, {"y": 1}, "yp -> y"),

		# Convert final c to l and return the lock
		rxn(medium, {"c": 1}, {"l": 1, "S": 1}, "c -> l + S"),
	]
	return result

def build_log_on_x_guarded(slow_clock=3e-4, faster=3e5, fast=3e3, medium=10.0):
	result = [
		rxn(slow_clock, {"S": 1, "b": 1, "x": 2}, {"a": 1, "b": 1, "x": 2}, "S+b+2x -> a+b+2x"),
		rxn(faster, {"a": 1, "x": 2}, {"c": 1, "xp": 1, "a": 1}, "a+2x -> c+xp+a"),
		rxn(faster, {"c": 2}, {"c": 1}, "2c -> c"),
		rxn(fast,   {"a": 1}, {}, "a -> 0"),
		rxn(medium, {"xp": 1}, {"x": 1}, "xp -> x"),
		rxn(medium, {"c": 1}, {"l": 1, "S": 1}, "c -> l + S"),
	]
	return result

def stop_log_y_done(state, t, steps):
	y = state.get("y", 0)
	return (y <= 1 and state.get("a", 0) == 0 and state.get("yp", 0) == 0 and state.get("c", 0) == 0 and state.get("S", 0) == 1)

def stop_log_x_done(state, t, steps):
	x = state.get("x", 0)
	return (x <= 1 and state.get("a", 0) == 0 and state.get("xp", 0) == 0 and state.get("c", 0) == 0 and state.get("S", 0) == 1)


def build_multiply(slow_x=1e-2, faster=3e5, fast=3e3, medium=10.0):
	"""
	Multiply module: (x, l) -> z
	"""
	return [
		rxn(slow_x, {"x": 1}, {"p": 1}, "x -> p"),
		rxn(faster, {"p": 1, "l": 1}, {"p": 1, "ls": 1, "zp": 1}, "p+l -> p+ls+zp"),
		rxn(fast,   {"p": 1}, {}, "p -> 0"),
		rxn(medium, {"ls": 1}, {"l": 1}, "ls -> l"),
		rxn(medium, {"zp": 1}, {"z": 1}, "zp -> z"),
	]

def stop_mult_done(state, t, steps):
	return (state.get("x", 0) == 0 and state.get("p", 0) == 0 and state.get("ls", 0) == 0 and state.get("zp", 0) == 0)

def build_exp(slow_l=1e-2, faster=3e5, fast=3e3, medium=10.0):
	"""
	Exp module: l -> y, starting from y=1
	"""
	return [
		rxn(slow_l, {"l": 1}, {"p": 1}, "l -> p"),
		rxn(faster, {"p": 1, "y": 1}, {"p": 1, "yp": 2}, "p+y -> p+2yp"),
		rxn(fast,   {"p": 1}, {}, "p -> 0"),
		rxn(medium, {"yp": 1}, {"y": 1}, "yp -> y"),
	]

def stop_exp_done(state, t, steps):
	return (state.get("l", 0) == 0 and state.get("p", 0) == 0 and state.get("yp", 0) == 0)


def run_problem_A_ssa_safe(X0, Y0, seed, params):
	# Phase 1: log(y) -> l
	log_rxns = build_log_on_y_guarded(
		slow_clock=params["slow_clock"],
		faster=params["faster"],
		fast=params["fast"],
		medium=params["medium"],
	)
	init1 = {"y": Y0, "b": 1, "S": 1}
	f1, t1, s1 = gillespie_ssa(log_rxns, init1, tmax=params["tmax"], seed=seed, stop_fn=stop_log_y_done)
	l = f1.get("l", 0)

	# Phase 2: multiply(x, l) -> z
	mult_rxns = build_multiply(
		slow_x=params["slow_x"],
		faster=params["faster"],
		fast=params["fast"],
		medium=params["medium"],
	)
	init2 = {"x": X0, "l": l}
	f2, t2, s2 = gillespie_ssa(mult_rxns, init2, tmax=params["tmax"], seed=seed + 999, stop_fn=stop_mult_done)
	return {"X0": X0, "Y0": Y0, "l": l, "z": f2.get("z", 0)}

def run_problem_B_ssa_safe(X0, seed, params):
	# Phase 1: log(x) -> l
	log_rxns = build_log_on_x_guarded(
		slow_clock=params["slow_clock"],
		faster=params["faster"],
		fast=params["fast"],
		medium=params["medium"],
	)
	init1 = {"x": X0, "b": 1, "S": 1}
	f1, t1, s1 = gillespie_ssa(log_rxns, init1, tmax=params["tmax"], seed=seed, stop_fn=stop_log_x_done)
	l = f1.get("l", 0)

	# Phase 2: exp(l) -> y
	exp_rxns = build_exp(
		slow_l=params["slow_x"],   # reuse the same slow scale
		faster=params["faster"],
		fast=params["fast"],
		medium=params["medium"],
	)
	init2 = {"l": l, "y": 1}
	f2, t2, s2 = gillespie_ssa(exp_rxns, init2, tmax=params["tmax"], seed=seed + 777, stop_fn=stop_exp_done)
	return {"X0": X0, "l": l, "y": f2.get("y", 0)}



# 5) Validation
def summarize_problem_A(testcases, params, nrep=60):
	rows = []
	for X0, Y0 in testcases:
		exp_l = int(round(math.log(Y0, 2)))
		exp_z = X0 * exp_l
		out = []
		for r in range(nrep):
			out.append(run_problem_A_ssa_safe(X0, Y0, seed=1000 + r * 11 + X0 + Y0, params=params))
		df = pd.DataFrame(out)
		rows.append({
			"X0": X0, "Y0": Y0,
			"expected_l": exp_l, "expected_z": exp_z,
			"l_mean": df["l"].mean(), "l_min": df["l"].min(), "l_max": df["l"].max(),
			"z_mean": df["z"].mean(), "z_min": df["z"].min(), "z_max": df["z"].max(),
			"ok_frac": ((df["l"] == exp_l) & (df["z"] == exp_z)).mean(),
		})
	return pd.DataFrame(rows)

def summarize_problem_B(testcases, params, nrep=60):
	rows = []
	for X0 in testcases:
		exp_l = int(round(math.log(X0, 2)))
		exp_y = X0
		out = []
		for r in range(nrep):
			out.append(run_problem_B_ssa_safe(X0, seed=2000 + r * 13 + X0, params=params))
		df = pd.DataFrame(out)
		rows.append({
			"X0": X0,
			"expected_l": exp_l, "expected_y": exp_y,
			"l_mean": df["l"].mean(), "l_min": df["l"].min(), "l_max": df["l"].max(),
			"y_mean": df["y"].mean(), "y_min": df["y"].min(), "y_max": df["y"].max(),
			"ok_frac": ((df["l"] == exp_l) & (df["y"] == exp_y)).mean(),
		})
	return pd.DataFrame(rows)



# Tuned parameters (guarded+locked log; faster >> fast >> medium >> slow_clock)
params = {
  "slow_clock": 3e-4,
  "slow_x": 1e-2,
  "medium": 10.0,
  "faster": 3e5,
  "fast": 3e3,
  "tmax": 2e7,
}

# Problem A tests
tests_A = [(3, 8), (5, 16), (2, 4), (7, 8)]
sA = summarize_problem_A(tests_A, params=params, nrep=1000)
print("\n=== SSA verification (Z = X0*log2(Y0))")
print(sA.to_string(index=False))

# Problem B tests
tests_B = [2, 4, 8, 16]
sB = summarize_problem_B(tests_B, params=params, nrep=1000)
print("\n=== SSA verification: (Y = 2^{log2(X0)})")
print(sB.to_string(index=False))
