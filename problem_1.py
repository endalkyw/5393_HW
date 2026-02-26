import numpy as np
import math

def firing_probability(x1, x2, x3):
    alpha_1 = 0.5 * x1 * (x1 - 1) * x2
    alpha_2 = x1 * x3 * (x3 - 1)
    alpha_3 = 3 * x2 * x3
    total = alpha_1 + alpha_2 + alpha_3
    return alpha_1/total, alpha_2/total, alpha_3/total

def simulate_hitting_probs(N=100, x0=(110, 26, 55), seed=None):
    if seed is not None:
        np.random.seed(seed)

    # Outcome conditions
    cond_1 = lambda x1, x2, x3: x1 >= 150
    cond_2 = lambda x1, x2, x3: x2 < 10
    cond_3 = lambda x1, x2, x3: x3 > 100

    outcome_counts = np.zeros(3, dtype=int)  # [C1, C2, C3]

    for sim in range(N):
        x1, x2, x3 = x0
        while True:
            # Check outcomes FIRST (First-passage race)
            if cond_1(x1, x2, x3):
                outcome_counts[0] += 1
                break
            if cond_2(x1, x2, x3):
                outcome_counts[1] += 1
                break
            if cond_3(x1, x2, x3):
                outcome_counts[2] += 1
                break
            p1, p2, p3 = firing_probability(x1, x2, x3)
            if (p1 + p2 + p3) == 0:
                break
            r = np.random.rand()
            if r < p1:
                x1 -= 2; x2 -= 1; x3 += 4
            elif r < p1 + p2:
                x1 -= 1; x3 -= 2; x2 += 3
            else:
                x2 -= 1; x3 -= 1; x1 += 2

            if sim%50 == 0: # print in every 50 simulations
                print(f"Sim. {sim}: x=({x1},{x2},{x3}), p=({p1:.3f},{p2:.4f},{p3:.4f})")
    print(f"Total Hits: C1={outcome_counts[0]}, C2={outcome_counts[1]}, C3={outcome_counts[2]}")
    return outcome_counts / N


## ----- part (a) ------
probs = simulate_hitting_probs(N=1000, x0=(110, 26, 55))
print("Estimated [Pr(C1), Pr(C2), Pr(C3)] =", probs)







## ----- part (b) ------
N = 100000
x0 = [9, 8, 7]
counts = []
for _ in range(N):
    x1, x2, x3 = x0
    for i in range(7):
        p1, p2, p3 = firing_probability(x1, x2, x3)
        r = np.random.rand()
        if r < p1:
            x1 -= 2; x2 -= 1; x3 += 4
        elif r < p1 + p2:
            x1 -= 1; x3 -= 2; x2 += 3
        else:
            x2 -= 1; x3 -= 1; x1 += 2

    counts.append([x1, x2, x3])


counts = np.array(counts)
mean = counts.mean(axis = 0)
var  = counts.var(axis = 0)
print("mean:", mean)
print("variance:", var)