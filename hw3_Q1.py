import numpy as np
from math import comb


def randomsynthesizer(probability, length, rng=None):
    if not (0.0 <= probability <= 1.0):
        raise ValueError("probability must be between 0 and 1")

    if rng is None:
        rng = np.random.default_rng()

    return (rng.random(length) < probability).astype(np.uint8)


def counter(vectors):
    arr = np.asarray(vectors, dtype=np.uint8)

    if arr.ndim != 2:
        raise ValueError("vectors must be a 2D array-like object")

    return arr.sum(axis=0).astype(np.uint8)


def mux(input_vectors, selector_series):
    inputs = np.asarray(input_vectors, dtype=np.uint8)
    selectors = np.asarray(selector_series, dtype=np.int64)

    if inputs.ndim != 2:
        raise ValueError("input_vectors must be 2D with shape (num_inputs, length)")

    num_inputs, length = inputs.shape

    if selectors.ndim != 1:
        raise ValueError("selector_series must be 1D")

    if len(selectors) != length:
        raise ValueError("selector_series must have same length as input bitstreams")

    if selectors.min() < 0 or selectors.max() >= num_inputs:
        raise ValueError("selector value out of range for given number of inputs")

    return inputs[selectors, np.arange(length)]


def eval_power_polynomial(a_coeffs, t):
    return sum(a * (t ** i) for i, a in enumerate(a_coeffs))


def eval_bernstein_polynomial(b_coeffs, t):
    n = len(b_coeffs) - 1
    return sum(
        b_coeffs[i] * comb(n, i) * (t ** i) * ((1 - t) ** (n - i))
        for i in range(n + 1)
    )


def synthesize_bernstein_mux(b_coeffs, t, length=100000, seed=None):
    if not (0.0 <= t <= 1.0):
        raise ValueError("t must be between 0 and 1")

    rng = np.random.default_rng(seed)
    b_coeffs = np.asarray(b_coeffs, dtype=float)

    # Degree n polynomial => n independent t-streams, and n+1 mux inputs
    n = len(b_coeffs) - 1

    # Generate coefficient input streams
    input_streams = np.vstack([
        randomsynthesizer(p, length, rng) for p in b_coeffs
    ])

    # Generate n independent t-streams
    # IMPORTANT: these should be independent streams with the same probability t
    t_streams = np.vstack([
        randomsynthesizer(t, length, rng) for _ in range(n)
    ])

    # Count number of ones -> selector values in {0,1,...,n}
    selector_series = counter(t_streams)
    # MUX output
    output_stream = mux(input_streams, selector_series)

    return {
        "input_streams": input_streams,
        "t_streams": t_streams,
        "selector_series": selector_series,
        "output_stream": output_stream,
        "empirical_output_probability": output_stream.mean(),
        "theoretical_bernstein_value": eval_bernstein_polynomial(b_coeffs, t),
    }


def problem1_test(t = 2/3):
    length = 1000

    # Original polynomial in power basis:
    # f(t) = 0 + 1*t + (-1/4)*t^2
    a_coeffs = [0.0, 1.0, -0.25]

    # Bernstein coefficients
    b_coeffs = [0.0, 0.5, 0.75]

    results = synthesize_bernstein_mux(
        b_coeffs=b_coeffs,
        t=t,
        length=length,
        seed=123
    )

    empirical = results["empirical_output_probability"]
    theoretical_bernstein = results["theoretical_bernstein_value"]
    theoretical_power = eval_power_polynomial(a_coeffs, t)

    print("Problem 1: f(t) = t - t^2/4 ------------------------------ ")
    print("t =", t)
    print("Bitstream length =", length)
    print("Theoretical value from power polynomial    :", theoretical_power)
    print("Theoretical value from Bernstein polynomial:", theoretical_bernstein)
    print("Empirical value from stochastic simulation :", empirical)
    print("Absolute error vs polynomial               :", abs(empirical - theoretical_power))


def problem2_test(t = 2/3):
    length = 1000

    # 4th-order Taylor approximation of cos(t):
    # f(t) = 1 - t^2/2 + t^4/24
    a_coeffs = [1.0, 0.0, -0.5, 0.0, 1.0 / 24.0]

    # Bernstein coefficients
    b_coeffs = [1.0, 1.0, 11.0 / 12.0, 3.0 / 4.0, 13.0 / 24.0]

    results = synthesize_bernstein_mux(
        b_coeffs=b_coeffs,
        t=t,
        length=length,
        seed=123
    )

    empirical = results["empirical_output_probability"]
    theoretical_bernstein = results["theoretical_bernstein_value"]
    theoretical_power = eval_power_polynomial(a_coeffs, t)

    print("Problem 2: cos(t) ≈ 1 - t^2/2 + t^4/24 -------------------")
    print("t =", t)
    print("Bitstream length =", length)
    print("Theoretical value from power polynomial    :", theoretical_power)
    print("Theoretical value from Bernstein polynomial:", theoretical_bernstein)
    print("Empirical value from stochastic simulation :", empirical)
    print("Absolute error vs polynomial               :", abs(empirical - theoretical_power))

def problem3_test(t = 2/3):
    length = 2000

    # Problem 3 polynomial:
    # f(t) = (31/32)t^5 + (5/32)t^4 - (5/8)t^3 + (5/4)t^2 - (5/4)t + 1/2
    a_coeffs = [0.5, -5.0 / 4.0, 5.0 / 4.0, -5.0 / 8.0, 5.0 / 32.0, 31.0 / 32.0]

    # Bernstein coefficients
    b_coeffs = [0.5, 0.25, 0.125, 1.0 / 16.0, 1.0 / 32.0, 1.0]

    results = synthesize_bernstein_mux(
        b_coeffs=b_coeffs,
        t=t,
        length=length,
        seed=123
    )

    empirical = results["empirical_output_probability"]
    theoretical_bernstein = results["theoretical_bernstein_value"]
    theoretical_power = eval_power_polynomial(a_coeffs, t)

    print("Problem 3: fifth-order polynomial ------------------------")
    print("t =", t)
    print("Bitstream length =", length)
    print("Theoretical value from power polynomial    :", theoretical_power)
    print("Theoretical value from Bernstein polynomial:", theoretical_bernstein)
    print("Empirical value from stochastic simulation :", empirical)
    print("Absolute error vs polynomial               :", abs(empirical - theoretical_power))



t = [0, 0.25, 0.5, 0.75, 1.0]

for ti in t:
    print(f"\nTesting t = {ti}")
    problem1_test(t=ti)
    problem2_test(t=ti)
    problem3_test(t=ti)