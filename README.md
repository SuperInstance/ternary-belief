# ternary-belief

**Belief propagation on ternary factor graphs: sum-product message passing with {-1, 0, +1} variables.**

Factor graphs are the workhorse of probabilistic inference. Each variable takes values in {-1, 0, +1}, and factors encode compatibility constraints between variables. The sum-product algorithm passes messages (3-vectors) along edges until beliefs converge.

The ternary setting is natural: each message is a distribution `[P(-1), P(0), P(+1)]`, and the 0 state carries "I don't know" uncertainty that helps convergence. Unlike binary belief propagation where disagreement is absolute, ternary variables can encode partial agreement, neutrality, and graded confidence — three states that map naturally to human decision-making (reject / abstain / accept).

---

## Motivation

Why ternary belief propagation instead of binary or continuous?

**Binary BP** is elegant but rigid: variables are forced into {0, 1} with no middle ground. In many real-world scenarios — sensor fusion, consensus protocols, expert systems — the most honest state is "uncertain." Ternary variables naturally encode this via the 0 state.

**Continuous BP** (e.g., Gaussian) is expressive but expensive. Ternary keeps the discrete simplicity while adding just one extra state, increasing per-message cost from O(4) to O(9) — a 2.25× cost for significantly richer expressivity.

**Applications** where ternary BP shines:
- **Sensor networks**: {-1, 0, +1} = {failure, unknown, ok}
- **Consensus protocols**: {disagree, abstain, agree}
- **Active inference**: policies evaluated as {avoid, explore, exploit}
- **Error correction**: ternary LDPC codes over Z₃
- **Decision systems**: reject / defer / accept classifications

---

## Architecture

The crate provides a complete loopy belief propagation engine built around three core types:

### `VariableNode`

Represents a random variable taking values in {-1, 0, +1}. Each variable maintains:
- `belief: [f64; 3]` — current marginal distribution `[P(-1), P(0), P(+1)]`
- `fixed: Option<i8>` — if `Some(v)`, the variable is evidence-clamped to value `v`
- `id: usize` — unique identifier in the factor graph

Key methods:
- `new(id)` — create with uniform prior `[1/3, 1/3, 1/3]`
- `fixed_val(id, val)` — create as observed evidence
- `normalize()` — renormalize beliefs to sum to 1.0
- `map()` — maximum a posteriori (MAP) estimate: argmax over {-1, 0, +1}
- `entropy()` — Shannon entropy in bits: `-Σ pᵢ log₂(pᵢ)`

### `PairFactor`

A pairwise factor connecting two ternary variables. Encoded as a 3×3 compatibility matrix:

```
           v1=-1   v1=0   v1=+1
v0=-1  [  compat[0][0]  compat[0][1]  compat[0][2]  ]
v0=0   [  compat[1][0]  compat[1][1]  compat[1][2]  ]
v0=+1  [  compat[2][0]  compat[2][1]  compat[2][2]  ]
```

Compatibility values are unnormalized potentials: higher means the configuration is more likely. A value of 0.0 forbids the configuration entirely; 1.0 is neutral; values > 1.0 encourage the pairing.

Methods:
- `new(v0, v1)` — create with uniform compatibility (all 1.0)
- `set(a, b, val)` / `get(a, b)` — accessor with ternary index translation

### `TernaryFactorGraph`

The inference engine. Maintains a bipartite graph of variables and pairwise factors, implements the sum-product algorithm.

```
┌─────────────────────────────────────────┐
│         TernaryFactorGraph              │
│  ┌─────────────┐     ┌─────────────┐   │
│  │ Variable 0  │◄───►│  Factor 0   │   │
│  │ [P(-1,0,+1)]│     │ 3×3 compat  │   │
│  └─────────────┘     └─────────────┘   │
│  ┌─────────────┐     ┌─────────────┐   │
│  │ Variable 1  │◄───►│  Factor 1   │   │
│  │ [P(-1,0,+1)]│     │ 3×3 compat  │   │
│  └─────────────┘     └─────────────┘   │
│         ...                ...          │
└─────────────────────────────────────────┘
```

Core methods:
- `add_variable()` / `add_fixed_variable(val)` — create free or observed variables
- `add_pair_factor(v0, v1)` — add a pairwise compatibility constraint
- `bp_round()` — one synchronous round of sum-product message passing
- `run_loopy_bp(n)` — iterate `bp_round()` n times
- `marginal(vid)` — return current belief distribution for variable `vid`
- `map_assignment()` — return the MAP configuration for all variables
- `energy(assignment)` — compute `-Σ log(compatibility)` (lower = better)

### `set_evidence()`

Helper function to clamp a variable to an observed value after graph construction. Re-applied automatically after every BP round to maintain evidence consistency.

---

## How It Works: Sum-Product Message Passing

Belief propagation operates by exchanging messages between variables and factors. For a pairwise factor `f` connecting variables `v0` and `v1`:

**Message from factor to v0** (summing over v1):
```
msg_f→v0(a) = Σ_{b ∈ {-1,0,+1}}  compatibility(a, b) × belief_v1(b)
```

**Message from factor to v1** (summing over v0):
```
msg_f→v1(b) = Σ_{a ∈ {-1,0,+1}}  compatibility(a, b) × belief_v0(a)
```

After all messages are computed, each variable updates its belief by multiplying all incoming messages and normalizing. Evidence-clamped variables override this with a delta distribution.

This is the **loopy BP** algorithm: on graphs with cycles (which are common), convergence is not guaranteed, but in practice it works well for many problems. Each round is O(|E| × 9) where |E| is the number of edges (pair factors).

### Why Ternary Messages Help Convergence

The 0-state acts as an information buffer. In binary BP, a conflicting message forces a hard choice immediately. In ternary BP, uncertainty can flow through the 0-state, allowing the network to find consistent configurations more gracefully. This is particularly visible in chain propagation (see tests): evidence at one end reliably propagates to the other.

---

## Usage Examples

### Basic Agreement Factor

Create two variables connected by a factor that strongly prefers them to agree:

```rust
use ternary_belief::{TernaryFactorGraph, set_evidence};

let mut graph = TernaryFactorGraph::new();
let v0 = graph.add_variable();
let v1 = graph.add_variable();

let f = graph.add_pair_factor(v0, v1);
// Strongly prefer same values; penalize disagreement
for a in -1..=1i8 {
    for b in -1..=1i8 {
        graph.factors[f].set(a, b, if a == b { 10.0 } else { 0.1 });
    }
}

// Observe v0 = +1
set_evidence(&mut graph, v0, 1);
graph.run_loopy_bp(5);

assert_eq!(graph.variables[v1].map(), 1);  // v1 agrees!
let marginal = graph.marginal(v1);
println!("P(v1=-1) = {:.3}", marginal[0]);
println!("P(v1=0)  = {:.3}", marginal[1]);
println!("P(v1=+1) = {:.3}", marginal[2]);
```

### Chain Propagation

Propagate evidence through a chain of variables:

```rust
let mut g = TernaryFactorGraph::new();
let v0 = g.add_variable();
let v1 = g.add_variable();
let v2 = g.add_variable();

let f0 = g.add_pair_factor(v0, v1);
let f1 = g.add_pair_factor(v1, v2);

// Agreement factors along the chain
for f in &[f0, f1] {
    for a in -1..=1i8 {
        for b in -1..=1i8 {
            g.factors[*f].set(a, b, if a == b { 10.0 } else { 0.1 });
        }
    }
}

set_evidence(&mut g, v0, -1);
g.run_loopy_bp(5);

// All beliefs should be valid probability distributions
for v in &g.variables {
    let sum: f64 = v.belief.iter().sum();
    assert!((sum - 1.0).abs() < 0.01);
}
println!("MAP assignment: {:?}", g.map_assignment());
```

### Energy Evaluation

Compare the energy of different assignments:

```rust
let mut g = TernaryFactorGraph::new();
g.add_variable();
g.add_variable();
let f = g.add_pair_factor(0, 1);
g.factors[f].set(1, 1, 10.0);   // strongly favor (+1, +1)
g.factors[f].set(-1, 1, 0.1);   // penalize (-1, +1)

let energy_agree = g.energy(&[1, 1]);
let energy_disagree = g.energy(&[-1, 1]);
assert!(energy_agree < energy_disagree);
println!("Energy [1,1] = {:.2}, Energy [-1,1] = {:.2}",
         energy_agree, energy_disagree);
```

### Entropy and Uncertainty

```rust
let v = VariableNode::new(0);
println!("Uniform entropy: {:.3} bits", v.entropy());  // ≈ 1.585

let d = VariableNode::fixed_val(0, 1);
println!("Deterministic entropy: {:.3} bits", d.entropy());  // ≈ 0.0
```

---

## Mathematical Background

### The Sum-Product Algorithm

Belief propagation, also known as the sum-product algorithm, computes marginal distributions for variables in a graphical model (Pearl 1988). For a factor graph with variables X and factors F, the marginal of variable xᵢ is:

```
P(xᵢ) ∝ Π_{f ∈ neighbors(xᵢ)}  μ_{f→xᵢ}(xᵢ)
```

where the factor-to-variable message is:

```
μ_{f→xᵢ}(xᵢ) = Σ_{X_f \ {xᵢ}}  f(X_f)  Π_{xⱼ ∈ neighbors(f) \ {xᵢ}}  μ_{xⱼ→f}(xⱼ)
```

For pairwise factors, this reduces to the matrix-vector products implemented in `bp_round()`.

### Ternary vs. Binary Complexity

| Operation | Binary {0,1} | Ternary {-1,0,+1} | Ratio |
|-----------|-------------|-------------------|-------|
| Message size | 2 | 3 | 1.5× |
| Per-message ops | 4 multiplies + 2 adds | 9 multiplies + 6 adds | ~2.25× |
| Expressivity | 2 states | 3 states (incl. neutral) | 1.5× |

The 0-state is not "wasted" information — it carries meaningful uncertainty that improves convergence on graphs with conflicting evidence.

### Energy and MAP

The energy (or "score") of an assignment is:

```
E(x) = - Σ_{factors f} log( compatibility_f(x_{v0}, x_{v1}) )
```

Lower energy means the assignment better satisfies all factor constraints. The MAP assignment returned by `map_assignment()` greedily selects the highest-belief state per variable, which is not guaranteed to be the global energy minimum but is a good approximation.

### Loopy BP Convergence

On tree-structured graphs, BP converges to exact marginals in finite time. On loopy graphs, convergence is not guaranteed (Murphy, Weiss & Jordan 1999). Practical strategies:
- Run for a fixed number of iterations (5–50)
- Check for belief oscillation
- Use damping (not yet implemented; see Future Work)

---

## Research Connections

### Active Inference and the Free Energy Principle

Ternary belief propagation is a natural inference engine for **active inference** (Friston 2010). In active inference, agents minimize variational free energy by updating beliefs about hidden states and selecting policies. Ternary hidden states map cleanly to:
- `-1` = the world is unfavorable (avoid)
- `0` = insufficient information (explore)
- `+1` = the world is favorable (exploit)

The companion crate `ternary-active-inference` builds on this foundation.

### Ternary Error-Correcting Codes

Belief propagation is the decoding algorithm for LDPC codes. Ternary LDPC codes over Z₃ (the integers modulo 3) are an active research area in coding theory, offering better performance than binary codes for certain channel models.

### Sensor Fusion and Distributed Consensus

In sensor networks with unreliable nodes, the 0-state naturally represents "sensor offline" or "no data." BP on ternary factor graphs provides a principled way to fuse noisy ternary observations into a coherent global picture. See `ternary-consensus` and `ternary-quorum` in the ecosystem.

### Rock-Paper-Scissors and Cyclic Dynamics

The {-1, 0, +1} state space with cyclic dominance (where -1 beats +1, +1 beats 0, 0 beats -1) models non-transitive competition. Belief propagation on such compatibility matrices can analyze the stationary distributions of cyclic ecological and game-theoretic systems (Reichenbach, Mobilia & Frey 2008).

### Connection to Z₃ Group Theory

The ternary values form the cyclic group Z₃ under addition modulo 3. Compatibility matrices that respect this group structure (e.g., `compat[a,b] = f(a - b mod 3)`) have special spectral properties that can accelerate convergence. This group-theoretic perspective connects to harmonic analysis on finite groups.

---

## Ecosystem

- **ternary-free-energy** — Free energy computations for ternary systems
- **ternary-active-inference** — Active inference using belief propagation
- **ternary-consensus** — Consensus via message passing
- **ternary-quorum** — Quorum voting on ternary preferences
- **ternary-grad** — Gradient descent for training ternary networks
- **ternary-attention** — Attention mechanisms with ternary states

## License

MIT
