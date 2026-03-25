import warnings
import numpy as np
import math
import json
import os
import time
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from helper import *


# CRN mechanics
# ----------------------------
def compute_propensities(reactions, state):
    alphas = []
    for r in reactions.values():
        a = r["k"]
        for sp, sto in zip(r["c"], r["c_m"]):
            n = state.get(sp, 0)
            if n < sto:
                a = 0.0
                break
            a *= math.comb(n, sto)
        alphas.append(a)
    a0 = float(sum(alphas))
    return alphas, a0

def pick_reaction(alphas, a0, rng):
    if a0 <= 0:
        return None
    u = rng.random() * a0
    s = 0.0
    for j, a in enumerate(alphas):
        s += a
        if u <= s:
            return j
    return len(alphas) - 1

def apply_reaction(state, reaction):
    for sp, sto in zip(reaction["c"], reaction["c_m"]):
        state[sp] = state.get(sp, 0) - sto
        if state[sp] < 0:
            raise RuntimeError(f"Negative count for {sp}")
    for sp, sto in zip(reaction["p"], reaction["p_m"]):
        state[sp] = state.get(sp, 0) + sto

# IMPORTANT: for your assignment, you typically want ONLY cI2/Cro2 as stop conditions.
# But I'll keep the generic one AND also allow overriding with explicit stop names.
def check_stops_generic(initial_values, state):
    hits = []
    for name, meta in initial_values.items():
        if meta["cond"] == "GE" and meta["stop"] is not None:
            if state.get(name, 0) >= meta["stop"]:
                hits.append(name)
    return hits

def check_stops_explicit(state, stealth_name, stealth_thr, hijack_name, hijack_thr):
    # Use strict ">" as in your problem statement; change to ">=" if needed
    if state.get(stealth_name, 0) > stealth_thr:
        return stealth_name
    if state.get(hijack_name, 0) > hijack_thr:
        return hijack_name
    return None

# ----------------------------
# Gillespie SSA trajectory
# ----------------------------
def simulate_one_ssa(reactions, initial_values, MOI=None,
                     max_time=1e6, max_events=2_000_000, seed=None,
                     use_explicit_stops=True,
                     stealth_name="cI2", stealth_thr=145,
                     hijack_name="Cro2", hijack_thr=55):
    rng = np.random.default_rng(seed)

    state = {k: v["start"] for k, v in initial_values.items()}
    if MOI is not None and "MOI" in state:
        state["MOI"] = int(MOI)

    t = 0.0
    rxn_list = list(reactions.values())

    # check stop at t=0
    if use_explicit_stops:
        w0 = check_stops_explicit(state, stealth_name, stealth_thr, hijack_name, hijack_thr)
        if w0 is not None:
            return {"winner": w0, "time": 0.0, "events": 0, "state": state}
    else:
        hits0 = check_stops_generic(initial_values, state)
        if hits0:
            hits_sorted = sorted(hits0, key=lambda nm: initial_values[nm]["idx"])
            return {"winner": hits_sorted[0], "time": 0.0, "events": 0, "state": state}

    for ev in range(1, max_events + 1):
        alphas, a0 = compute_propensities(reactions, state)
        if a0 <= 0:
            return {"winner": None, "time": t, "events": ev - 1, "state": state}

        # SSA time step
        u1 = rng.random()
        dt = -math.log(u1) / a0
        t += dt
        if t > max_time:
            return {"winner": None, "time": t, "events": ev - 1, "state": state}

        j = pick_reaction(alphas, a0, rng)
        apply_reaction(state, rxn_list[j])

        if use_explicit_stops:
            w = check_stops_explicit(state, stealth_name, stealth_thr, hijack_name, hijack_thr)
            if w is not None:
                return {"winner": w, "time": t, "events": ev, "state": state}
        else:
            hits = check_stops_generic(initial_values, state)
            if hits:
                hits_sorted = sorted(hits, key=lambda nm: initial_values[nm]["idx"])
                return {"winner": hits_sorted[0], "time": t, "events": ev, "state": state}

    return {"winner": None, "time": t, "events": max_events, "state": state}

# ----------------------------
# Worker for one MOI
# ----------------------------
def run_one_moi(moi, r_path, in_path, N, seed, out_dir,
                max_time, max_events,
                stealth_name, stealth_thr, hijack_name, hijack_thr,
                use_explicit_stops=True,
                dump_each_moi=True):
    reactions = parse_dot_r_file(r_path)
    initial_values = parse_dot_in_file(in_path)

    rng = np.random.default_rng(seed + moi * 1000003)

    counts = {"stealth": 0, "hijack": 0, "none": 0, "other": 0}
    t_start = time.time()

    for i_n in range(N):
        s = int(rng.integers(0, 2**31 - 1))
        print(f"N: {i_n}, MOI: {moi} ----------------- ")
        res = simulate_one_ssa(
            reactions, initial_values, MOI=moi, seed=s,
            max_time=max_time, max_events=max_events,
            use_explicit_stops=use_explicit_stops,
            stealth_name=stealth_name, stealth_thr=stealth_thr,
            hijack_name=hijack_name, hijack_thr=hijack_thr,
        )

        w = res["winner"]
        if w is None:
            counts["none"] += 1
        elif w == stealth_name:
            counts["stealth"] += 1
        elif w == hijack_name:
            counts["hijack"] += 1
        else:
            counts["other"] += 1

    elapsed = time.time() - t_start
    result = {
        "MOI": int(moi),
        "N": int(N),
        "seed": int(seed),
        "max_time": float(max_time),
        "max_events": int(max_events),
        "stops": {
            "stealth": {"name": stealth_name, "thr": stealth_thr},
            "hijack": {"name": hijack_name, "thr": hijack_thr},
            "use_explicit_stops": bool(use_explicit_stops),
        },
        "counts": counts,
        "probs": {k: counts[k] / N for k in counts},
        "elapsed_sec": elapsed,
    }

    if dump_each_moi:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"results_moi_{moi}.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

    return result

# CLI (I used CLI to automate the whole program from a bash script externally)
# ----------------------------
def parse_moi_range(s):
    # accepts "1:10" or "1-10" or "1,2,3"
    s = s.strip()
    if "," in s:
        return [int(x) for x in s.split(",") if x.strip()]
    if ":" in s:
        a, b = s.split(":")
        a, b = int(a), int(b)
        return list(range(a, b + 1))
    if "-" in s:
        a, b = s.split("-")
        a, b = int(a), int(b)
        return list(range(a, b + 1))
    return [int(s)]

def main():
    # This CLI parser is created by ChatGPT -------------------------------------------------------
    ap = argparse.ArgumentParser(description="Lambda SSA Monte Carlo with MOI sweep + parallelism")
    ap.add_argument("--r", dest="r_path", default="lambda.r", help="Path to lambda.r")
    ap.add_argument("--in", dest="in_path", default="lambda.in", help="Path to lambda.in")
    ap.add_argument("--moi", type=int, default=None, help="Run a single MOI value")
    ap.add_argument("--moi-range", type=str, default="1:10", help='MOI sweep, e.g. "1:10" or "1-10" or "1,2,3"')
    ap.add_argument("--N", type=int, default=2000, help="Trajectories per MOI")
    ap.add_argument("--seed", type=int, default=123, help="Base RNG seed")
    ap.add_argument("--jobs", type=int, default=None, help="Number of parallel processes (default: cpu count)")
    ap.add_argument("--outdir", type=str, default="out_lambda", help="Directory to dump JSON results")
    ap.add_argument("--max-time", type=float, default=5e4, help="SSA max time (safety)")
    ap.add_argument("--max-events", type=int, default=200000, help="SSA max events (safety)")
    ap.add_argument("--stealth-name", type=str, default="cI2")
    ap.add_argument("--stealth-thr", type=int, default=145)
    ap.add_argument("--hijack-name", type=str, default="Cro2")
    ap.add_argument("--hijack-thr", type=int, default=55)

    ap.add_argument("--use-generic-stops", action="store_true",
                    help="If set, use ALL GE stops from .in instead of only cI2/Cro2 thresholds")

    args = ap.parse_args()

    if args.moi is not None:
        moi_values = [args.moi]
    else:
        moi_values = parse_moi_range(args.moi_range)

    use_explicit = not args.use_generic_stops
    os.makedirs(args.outdir, exist_ok=True)

    # Run in parallel across MOI values
    results_all = {}
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = []
        for moi in moi_values:
            futs.append(ex.submit(
                run_one_moi,
                moi, args.r_path, args.in_path, args.N, args.seed, args.outdir,
                args.max_time, args.max_events,
                args.stealth_name, args.stealth_thr, args.hijack_name, args.hijack_thr,
                use_explicit, True
            ))

        for fut in as_completed(futs):
            res = fut.result()
            results_all[str(res["MOI"])] = res
            print(f"[done] MOI={res['MOI']}  "
                  f"P(stealth)={res['probs']['stealth']:.4f}  "
                  f"P(hijack)={res['probs']['hijack']:.4f}  "
                  f"P(none)={res['probs']['none']:.4f}  "
                  f"elapsed={res['elapsed_sec']:.1f}s")

            # Update combined file as we go (so you always have partial progress)
            combined_path = os.path.join(args.outdir, "results_all.json")
            with open(combined_path, "w") as f:
                json.dump(results_all, f, indent=2)

    print(f"\nWrote per-MOI JSON to: {args.outdir}/results_moi_*.json")
    print(f"Wrote combined JSON to: {args.outdir}/results_all.json")


if __name__ == "__main__":
    # main()
    visualize_result("out_lambda/results_all.json")