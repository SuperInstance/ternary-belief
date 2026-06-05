# ternary-belief

Belief propagation on ternary networks with variables in `{-1, 0, +1}`.

## Components

- `FactorGraph` — bipartite graph of variable and factor nodes
- `Variable` — ternary variable node with optional evidence clamping
- `Factor` — joint probability table over its variables
- `MessagePassing` — sum-product variable→factor and factor→variable messages
- `LoopyBeliefPropagation` — iterates until convergence with tolerance check

## Usage

```rust
use ternary_belief::{FactorGraph, LoopyBeliefPropagation};

let mut g = FactorGraph::new();
let x0 = g.add_variable("x0");
let x1 = g.add_variable("x1");
g.add_factor(vec![x0, x1], vec![/* 9 values */]);
g.clamp_evidence(x0, -1);

let lbp = LoopyBeliefPropagation::new(100, 1e-6);
let marginals = lbp.marginals(&g);
```
