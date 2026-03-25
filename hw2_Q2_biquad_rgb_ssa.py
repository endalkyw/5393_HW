from __future__ import annotations
import math
import random
from math import comb
from typing import Dict, List, Tuple

SPECIES = ["X", "A", "Y", "C", "F", "E", "H", "R1", "G1", "B1", "R2", "G2", "B2"]
IDX = {name: i for i, name in enumerate(SPECIES)}

Reaction = Tuple[Dict[str, int], Dict[str, int], float, str]
State = List[int]


def rxn(reactants: Dict[str, int], products: Dict[str, int], rate: float, name: str) -> Reaction:
    return reactants, products, rate, name


MULTIPLIER_REACTIONS: List[Reaction] = [
    rxn({"A": 8}, {"Y": 1}, 10.0, "8A->Y"),
    rxn({"C": 8}, {"Y": 1}, 10.0, "8C->Y"),
    rxn({"E": 8}, {"Y": 1}, 10.0, "8E->Y"),
    rxn({"F": 8}, {"X": 1}, 10.0, "8F->X"),
    rxn({"H": 8}, {"X": 1}, 10.0, "8H->X"),
]

PHASE_REACTIONS: Dict[str, List[Reaction]] = {
    "g": [
        rxn({"X": 1}, {"R1": 1, "A": 1}, 1.0, "X->R1+A"),
        rxn({"B1": 1}, {"R2": 1, "F": 1, "C": 1}, 1.0, "B1->R2+F+C"),
        rxn({"B2": 1}, {"H": 1, "E": 1}, 1.0, "B2->H+E"),
    ] + MULTIPLIER_REACTIONS,
    "b": [
        rxn({"R1": 1}, {"G1": 1}, 1.0, "R1->G1"),
        rxn({"R2": 1}, {"G2": 1}, 1.0, "R2->G2"),
    ] + MULTIPLIER_REACTIONS,
    "r": [
        rxn({"G1": 1}, {"B1": 1}, 1.0, "G1->B1"),
        rxn({"G2": 1}, {"B2": 1}, 1.0, "G2->B2"),
    ] + MULTIPLIER_REACTIONS,
}


def propensity(state: State, reactants: Dict[str, int], rate: float) -> float:
    """Mass-action propensity for a reaction channel."""
    value = rate
    for species, coeff in reactants.items():
        n = state[IDX[species]]
        if n < coeff:
            return 0.0
        value *= comb(n, coeff)
    return value


def fire_reaction(state: State, reactants: Dict[str, int], products: Dict[str, int]) -> State:
    """Apply a single reaction event."""
    new_state = state[:]
    for species, coeff in reactants.items():
        new_state[IDX[species]] -= coeff
    for species, coeff in products.items():
        new_state[IDX[species]] += coeff
    return new_state


def gillespie_phase(state: State, reactions: List[Reaction], rng: random.Random) -> Tuple[State, float, int]:
    """Run Gillespie SSA until no reaction in the current phase is enabled."""
    t = 0.0
    events = 0
    while True:
        props = [propensity(state, reactants, rate) for reactants, _, rate, _ in reactions]
        a0_sum = sum(props)
        if a0_sum == 0.0:
            return state, t, events

        u1 = rng.random()
        u2 = rng.random()
        tau = -math.log(u1) / a0_sum
        t += tau

        threshold = u2 * a0_sum
        running = 0.0
        chosen_index = 0
        for i, a in enumerate(props):
            running += a
            if running >= threshold:
                chosen_index = i
                break

        reactants, products, _, _ = reactions[chosen_index]
        state = fire_reaction(state, reactants, products)
        events += 1


def simulate_cycles(inputs: List[int], seed: int = 1) -> Tuple[List[int], State, float, int]:
    """Run the five-cycle RGB simulation.

    Each cycle:
      1) add the new input value to X,
      2) simulate g, then b, then r,
      3) sample Y,
      4) reset Y to zero,
      5) carry all other species to the next cycle.
    """
    rng = random.Random(seed)
    state: State = [0] * len(SPECIES)
    sampled_outputs: List[int] = []
    total_time = 0.0
    total_events = 0

    for x in inputs:
        state[IDX["X"]] += x
        for phase_name in ("g", "b", "r"):
            state, phase_time, phase_events = gillespie_phase(state, PHASE_REACTIONS[phase_name], rng)
            total_time += phase_time
            total_events += phase_events

        sampled_outputs.append(state[IDX["Y"]])
        state[IDX["Y"]] = 0  # sample/reset output

    return sampled_outputs, state, total_time, total_events


def format_state(state: State) -> Dict[str, int]:
    """Return a compact dictionary of nonzero species counts."""
    return {name: state[i] for i, name in enumerate(SPECIES) if state[i] != 0}


def run_demo() -> None:
    """Demonstrate the requested five-cycle input sequence."""
    inputs = [100, 5, 500, 20, 250]
    outputs, final_state, sim_time, events = simulate_cycles(inputs, seed=1)

    print("Input sequence:", inputs)
    print("Sampled Y:     ", outputs)
    print(f"SSA time:      {sim_time:.6f}")
    print(f"Events:        {events}")
    print("Final carried state:")
    for species, count in format_state(final_state).items():
        print(f"  {species} = {count}")


if __name__ == "__main__":
    run_demo()
