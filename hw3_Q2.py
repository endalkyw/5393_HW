from fractions import Fraction
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


# ============================================================
# Utility formatting
# ============================================================
def fmt_prob(x, ndigits=6):
    return f"{float(x):.{ndigits}f}"


def gate_name(gate, s):
    p = float(s)
    if gate == "AND":
        return f"A_{p:.1f}"
    elif gate == "NAND":
        return f"N_{p:.1f}"
    raise ValueError(f"Unknown gate type: {gate}")


def source_label(prob, idx):
    p = float(prob)
    if abs(p - 0.4) < 1e-12:
        return f"S04_{idx}"
    elif abs(p - 0.5) < 1e-12:
        return f"S05_{idx}"
    elif abs(p - 0.0) < 1e-12:
        return f"S00_{idx}"
    elif abs(p - 1.0) < 1e-12:
        return f"S10_{idx}"
    return f"S{str(p).replace('.', '')}_{idx}"


def leaf_label(prob):
    p = float(prob)
    if abs(p - 0.4) < 1e-12:
        return "LEAF_04"
    elif abs(p - 0.5) < 1e-12:
        return "LEAF_05"
    elif abs(p - 0.0) < 1e-12:
        return "LEAF_00"
    elif abs(p - 1.0) < 1e-12:
        return "LEAF_10"
    return f"LEAF_{str(p).replace('.', '')}"

def synthesize_exact_bfs(target_str, source_strs=("0.4", "0.5"), max_depth=40):
    target = Fraction(target_str)
    sources = [Fraction(s) for s in source_strs]
    leaves = {Fraction(0), Fraction(1), *sources}

    queue = deque([(target, [])])
    visited = {target}

    while queue:
        p, chain = queue.popleft()

        if p in leaves:
            return chain, p

        if len(chain) >= max_depth:
            continue

        for s in sources:
            q_and = p / s
            if Fraction(0) <= q_and <= Fraction(1) and q_and not in visited:
                visited.add(q_and)
                queue.append((q_and, chain + [("AND", s, q_and)]))

            q_nand = (Fraction(1) - p) / s
            if Fraction(0) <= q_nand <= Fraction(1) and q_nand not in visited:
                visited.add(q_nand)
                queue.append((q_nand, chain + [("NAND", s, q_nand)]))

    return None, None


# ============================================================
# Logic conversion
# ============================================================
def sequence_string(chain):
    return " -> ".join(gate_name(gate, s) for gate, s, _ in chain)


def build_logic_expression(chain, leaf):
    count04 = 0
    count05 = 0
    src_names = []

    for gate, s, _ in chain:
        if float(s) == 0.4:
            count04 += 1
            src_names.append(source_label(s, count04))
        elif float(s) == 0.5:
            count05 += 1
            src_names.append(source_label(s, count05))
        else:
            src_names.append(source_label(s, len(src_names) + 1))

    expr = leaf_label(leaf)

    for (gate, s, _), src in zip(reversed(chain), reversed(src_names)):
        if gate == "AND":
            expr = f"AND({src}, {expr})"
        elif gate == "NAND":
            expr = f"NOT(AND({src}, {expr}))"
        else:
            raise ValueError(f"Unknown gate {gate}")

    return expr


def build_gate_netlist(chain, leaf):
    count04 = 0
    count05 = 0
    src_names = []

    for gate, s, _ in chain:
        if float(s) == 0.4:
            count04 += 1
            src_names.append(source_label(s, count04))
        elif float(s) == 0.5:
            count05 += 1
            src_names.append(source_label(s, count05))
        else:
            src_names.append(source_label(s, len(src_names) + 1))

    lines = []
    current = leaf_label(leaf)
    net_id = 1

    for (gate, s, _), src in zip(reversed(chain), reversed(src_names)):
        if gate == "AND":
            out = f"n{net_id}"
            lines.append((out, "AND", src, current))
            current = out
            net_id += 1
        elif gate == "NAND":
            and_out = f"n{net_id}"
            lines.append((and_out, "AND", src, current))
            net_id += 1
            not_out = f"n{net_id}"
            lines.append((not_out, "NOT", and_out, None))
            current = not_out
            net_id += 1
        else:
            raise ValueError(f"Unknown gate {gate}")

    return lines, current


# ============================================================
# Probability evaluation
# ============================================================
def evaluate_chain_probability(chain, leaf):
    p = Fraction(leaf)
    for gate, s, _ in reversed(chain):
        if gate == "AND":
            p = s * p
        elif gate == "NAND":
            p = Fraction(1) - s * p
        else:
            raise ValueError(f"Unknown gate {gate}")
    return p


# ============================================================
# Bitstream generation and simulation
# ============================================================
def random_bitstream(probability, length, rng):
    return (rng.random(length) < probability).astype(np.uint8)


def leaf_bitstream(leaf, length, rng):
    p = float(leaf)
    if abs(p - 0.0) < 1e-12:
        return np.zeros(length, dtype=np.uint8)
    if abs(p - 1.0) < 1e-12:
        return np.ones(length, dtype=np.uint8)
    return random_bitstream(p, length, rng)


def simulate_chain(chain, leaf, length=10000, seed=123):
    rng = np.random.default_rng(seed)
    current = leaf_bitstream(leaf, length, rng)

    for gate, s, _ in reversed(chain):
        src = random_bitstream(float(s), length, rng)
        if gate == "AND":
            current = np.bitwise_and(src, current)
        elif gate == "NAND":
            current = 1 - np.bitwise_and(src, current)
        else:
            raise ValueError(f"Unknown gate {gate}")

    return current, current.mean()


# ============================================================
# Network visualization of actual gate-level logic
# ============================================================
def build_networkx_from_netlist(netlist, final_node, leaf):
    G = nx.DiGraph()

    # Add leaf node
    leaf_node = leaf_label(leaf)
    leaf_prob = float(leaf)
    G.add_node(leaf_node, kind="leaf", label=f"{leaf_node}\np={leaf_prob:.3f}")

    for out, gate_type, in1, in2 in netlist:
        if gate_type == "AND":
            gate_node = f"{out}_gate"
            G.add_node(gate_node, kind="gate_and", label="AND")
            G.add_node(out, kind="net", label=out)

            # Source or net nodes
            if in1 not in G:
                kind = "source" if in1.startswith("S") else "net"
                G.add_node(in1, kind=kind, label=in1)
            if in2 not in G:
                kind = "source" if in2.startswith("S") else "net"
                G.add_node(in2, kind=kind, label=in2)

            G.add_edge(in1, gate_node)
            G.add_edge(in2, gate_node)
            G.add_edge(gate_node, out)

        elif gate_type == "NOT":
            gate_node = f"{out}_gate"
            G.add_node(gate_node, kind="gate_not", label="NOT")
            G.add_node(out, kind="net", label=out)

            if in1 not in G:
                kind = "source" if in1.startswith("S") else "net"
                G.add_node(in1, kind=kind, label=in1)

            G.add_edge(in1, gate_node)
            G.add_edge(gate_node, out)

    # Final output alias node
    G.add_node("OUTPUT", kind="output", label="OUTPUT")
    G.add_edge(final_node, "OUTPUT")

    return G


def hierarchical_positions_from_netlist(netlist, final_node, leaf):
    """
    Make a left-to-right layout following the actual gate evaluation order.
    """
    positions = {}

    # Collect nodes by stage
    leaf_node = leaf_label(leaf)
    positions[leaf_node] = (0, 0)

    stage_x = 1
    y = 0
    touched_nodes = {leaf_node}

    for idx, (out, gate_type, in1, in2) in enumerate(netlist):
        gate_node = f"{out}_gate"

        # place inputs if new
        for inp in [in1] + ([] if in2 is None else [in2]):
            if inp not in positions:
                positions[inp] = (stage_x - 1, y - 1.5)
                y -= 1

        # gate and output
        positions[gate_node] = (stage_x, y)
        positions[out] = (stage_x + 0.6, y)
        y -= 1.5
        stage_x += 1

    positions["OUTPUT"] = (stage_x + 0.8, 0)
    return positions


def draw_gate_network(netlist, final_node, leaf, title="Synthesized AND/NOT Gate Network",
                      figsize=(18, 10)):
    G = build_networkx_from_netlist(netlist, final_node, leaf)
    pos = hierarchical_positions_from_netlist(netlist, final_node, leaf)

    plt.figure(figsize=figsize)

    node_labels = nx.get_node_attributes(G, "label")
    kinds = nx.get_node_attributes(G, "kind")

    source_nodes = [n for n, k in kinds.items() if k == "source"]
    leaf_nodes = [n for n, k in kinds.items() if k == "leaf"]
    net_nodes = [n for n, k in kinds.items() if k == "net"]
    and_nodes = [n for n, k in kinds.items() if k == "gate_and"]
    not_nodes = [n for n, k in kinds.items() if k == "gate_not"]
    output_nodes = [n for n, k in kinds.items() if k == "output"]

    nx.draw_networkx_nodes(G, pos, nodelist=source_nodes, node_shape="o", node_size=1800)
    nx.draw_networkx_nodes(G, pos, nodelist=leaf_nodes, node_shape="o", node_size=2200)
    nx.draw_networkx_nodes(G, pos, nodelist=net_nodes, node_shape="s", node_size=1300)
    nx.draw_networkx_nodes(G, pos, nodelist=and_nodes, node_shape="^", node_size=2200)
    nx.draw_networkx_nodes(G, pos, nodelist=not_nodes, node_shape="D", node_size=1800)
    nx.draw_networkx_nodes(G, pos, nodelist=output_nodes, node_shape="h", node_size=2400)

    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="-|>", arrowsize=18, width=1.6)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# ============================================================
# Full report + optional plot
# ============================================================
def report_synthesis_and_simulation(
    target_str,
    source_strs=("0.4", "0.5"),
    max_depth=40,
    length=100000,
    seed=123,
    plot_graph=True
):
    chain, leaf = synthesize_exact_bfs(target_str, source_strs=source_strs, max_depth=max_depth)

    if chain is None:
        print("=" * 78)
        print(f"Target probability: {target_str}")
        print("No exact synthesis found within the specified depth.")
        print("=" * 78)
        return None

    exact_target = Fraction(target_str)
    exact_eval = evaluate_chain_probability(chain, leaf)

    logic_expr = build_logic_expression(chain, leaf)
    netlist, final_node = build_gate_netlist(chain, leaf)

    output_stream, empirical_prob = simulate_chain(chain, leaf, length=length, seed=seed)

    print("=" * 78)
    print(f"Target probability: {target_str}")
    print(f"Sources: {source_strs}")
    print(f"Depth: {len(chain)}")
    print("-" * 78)

    for i, (gate, s, q_next) in enumerate(chain, start=1):
        ab = "(0,-1)" if gate == "AND" else "(1,1)"
        print(
            f"Level {i:2d}: gate={gate:4s}, (a,b)={ab}, "
            f"source={fmt_prob(s)}, next_target={fmt_prob(q_next)}"
        )

    print("-" * 78)
    print("Gate sequence:")
    print(sequence_string(chain))
    print()
    print("Nested AND/NOT combinational logic:")
    print(logic_expr)
    print()
    print("Gate-level netlist:")
    for out, gate_type, in1, in2 in netlist:
        if gate_type == "AND":
            print(f"{out} = AND({in1}, {in2})")
        else:
            print(f"{out} = NOT({in1})")
    print(f"Final output node: {final_node}")
    print("-" * 78)
    print("Verification:")
    print(f"Requested target probability        : {fmt_prob(exact_target, 9)}")
    print(f"Exact probability from gate chain   : {fmt_prob(exact_eval, 9)}")
    print(f"Simulated output probability        : {empirical_prob:.9f}")
    print(f"Absolute simulation error           : {abs(empirical_prob - float(exact_target)):.9f}")
    print(f"Bitstream length                    : {length}")
    print(f"Random seed                         : {seed}")
    print("=" * 78)
    print()

    if plot_graph:
        draw_gate_network(
            netlist,
            final_node,
            leaf,
            title=f"Gate Network for Target {target_str}"
        )

    return {
        "target": exact_target,
        "chain": chain,
        "leaf": leaf,
        "logic_expression": logic_expr,
        "netlist": netlist,
        "exact_probability": exact_eval,
        "empirical_probability": empirical_prob,
        "output_stream": output_stream,
        "final_node": final_node,
    }


# ============================================================
# Example usage
# ============================================================
if __name__ == "__main__":
    targets = ["0.8881188", "0.2119209", "0.5555555"]

    for t in targets:
        report_synthesis_and_simulation(
            target_str=t,
            source_strs=("0.4", "0.5"),
            max_depth=40,
            length=100000,
            seed=123,
            plot_graph=True
        )