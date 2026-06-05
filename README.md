# ternary-belief

**Belief propagation on ternary factor graphs: sum-product message passing with {-1, 0, +1} variables.**

Factor graphs are the workhorse of probabilistic inference. Each variable takes values in {-1, 0, +1}, and factors encode compatibility constraints between variables. The sum-product algorithm passes messages (3-vectors) along edges until beliefs converge.

The ternary setting is natural: each message is a distribution [P(-1), P(0), P(+1)], and the 0 state carries "I don't know" uncertainty that helps convergence.

---

## How It Works

```
Variable Nodes ←→ Factor Nodes

Variable → Factor message: product of all incoming factor messages
Factor → Variable message: sum over all other variables of compatibility × incoming

For ternary: each message is [p(-1), p(0), p(+1)]
```

The **sum-product rule** for a factor connecting variables X and Y:
```
msg_factor→X(a) = Σ_b compatibility(a,b) × msg_Y→factor(b)
                  for a,b ∈ {-1, 0, +1}
```

This is O(9) per message — 3× the cost of binary (O(4)) but with richer expressivity.

---

## Architecture

- **`VariableNode`** — Belief [P(-1), P(0), P(+1)], MAP estimate, entropy, evidence clamping
- **`PairFactor`** — 3×3 compatibility matrix between two ternary variables
- **`TernaryFactorGraph`** — Variables + factors with BP inference
- **`bp_round()`** — One round of sum-product message passing
- **`run_loopy_bp(n)`** — N iterations of loopy belief propagation
- **`set_evidence()`** — Clamp a variable to an observed value
- **`energy()`** — -Σ log(compatibility) under current assignment

---

## Quick Start

```rust
use ternary_belief::{TernaryFactorGraph, set_evidence};

let mut graph = TernaryFactorGraph::new();
let v0 = graph.add_variable();
let v1 = graph.add_variable();

let f = graph.add_pair_factor(v0, v1);
// Set high compatibility for agreement
for a in -1..=1i8 {
    for b in -1..=1i8 {
        graph.factors[f].set(a, b, if a == b { 10.0 } else { 0.1 });
    }
}

set_evidence(&mut graph, v0, 1); // observe v0 = +1
graph.run_loopy_bp(5);

let marginal = graph.marginal(v1);
println!("P(v1=+1) = {:.3}", marginal[2]);
```

---

## Ecosystem

- **ternary-free-energy** — Free energy computations for ternary systems
- **ternary-active-inference** — Active inference using belief propagation
- **ternary-consensus** — Consensus via message passing
- **ternary-quorum** — Quorum voting on ternary preferences

## License

MIT
