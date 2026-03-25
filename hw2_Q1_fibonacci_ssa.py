from __future__ import annotations
import math
import random
from math import comb
from typing import Dict, List, Tuple

State = Dict[str, int]
Reaction = Tuple[Dict[str, int], Dict[str, int]]


def build_fibonacci_reactions(num_updates: int = 11) -> List[Reaction]:
    """Build the unrolled finite Fibonacci reaction network."""
    reactions: List[Reaction] = []
    for i in range(num_updates):
        reactions.append(({f"A{i}": 1}, {f"B{i+1}": 1}))
        reactions.append(({f"B{i}": 1}, {f"A{i+1}": 1, f"B{i+1}": 1}))
    return reactions


def propensity(state: State, reactants: Dict[str, int], rate: float = 1.0) -> float:
    """Mass-action propensity for a single reaction channel."""
    value = rate
    for species, coeff in reactants.items():
        n = state.get(species, 0)
        if n < coeff:
            return 0.0
        value *= comb(n, coeff)
    return value


def fire_reaction(state: State, reactants: Dict[str, int], products: Dict[str, int]) -> State:
    """Apply one reaction event and return the updated state."""
    new_state = state.copy()

    for species, coeff in reactants.items():
        new_state[species] = new_state.get(species, 0) - coeff
        if new_state[species] == 0:
            del new_state[species]

    for species, coeff in products.items():
        new_state[species] = new_state.get(species, 0) + coeff

    return new_state


def gillespie_fibonacci(a0: int, b0: int, num_updates: int = 11, seed: int | None = 0) -> Tuple[State, float, int]:
    """Run Gillespie SSA until the finite network reaches completion.

    Returns
    -------
    final_state : dict
        Terminal molecular counts.
    time : float
        Simulated SSA time.
    events : int
        Number of reaction events executed.
    """
    rng = random.Random(seed)
    reactions = build_fibonacci_reactions(num_updates)
    state: State = {f"A0": a0, f"B0": b0}
    t = 0.0
    events = 0

    while True:
        propensities = [propensity(state, reactants) for reactants, _ in reactions]
        a0_sum = sum(propensities)
        if a0_sum == 0.0:
            return state, t, events

        u1 = rng.random()
        u2 = rng.random()
        tau = -math.log(u1) / a0_sum
        t += tau

        threshold = u2 * a0_sum
        running = 0.0
        chosen_index = 0
        for i, a in enumerate(propensities):
            running += a
            if running >= threshold:
                chosen_index = i
                break

        reactants, products = reactions[chosen_index]
        state = fire_reaction(state, reactants, products)
        events += 1


def extract_output(final_state: State, num_updates: int = 11) -> Tuple[int, int]:
    """Return (A_final, B_final)."""
    return final_state.get(f"A{num_updates}", 0), final_state.get(f"B{num_updates}", 0)


def run_demo() -> None:
    """Demonstrate the two requested initial conditions."""
    for initial in [(0, 1), (3, 7)]:
        final_state, sim_time, events = gillespie_fibonacci(*initial, num_updates=11, seed=1)
        afinal, bfinal = extract_output(final_state, num_updates=11)
        print(f"Initial state: A0={initial[0]}, B0={initial[1]}")
        print(f"Final state:   A11={afinal}, B11={bfinal}")
        print(f"SSA time:      {sim_time:.6f}")
        print(f"Events:        {events}")
        print()


if __name__ == "__main__":
    run_demo()
