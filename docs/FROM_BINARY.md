# From Binary to Ternary: Belief Propagation

## The Trap

Binary belief propagation models variables that take two states: true or false, on or off, 0 or 1. Messages are 2-vectors: [P(false), P(true)]. Factor graphs over binary variables are the standard tool for error-correcting codes, Bayesian networks, and constraint satisfaction.

But binary variables force a painful choice when the evidence is ambiguous: the posterior must land on one side or the other. There's no "I genuinely don't know" state that the message-passing can preserve. The variable will converge to a belief with P(true) ≈ 0.5 — high entropy — but the graph has no way to represent "this variable is undefined." The BP algorithm keeps iterating, searching for a resolution that may not exist.

## Map to Three States

| Domain | −1 | 0 | +1 |
|--------|----|---|-----|
| Variable value | false / negative | unknown / neutral | true / positive |
| Message | P(−1) | P(0) | P(+1) |
| Evidence | clamped to −1 | unobserved | clamped to +1 |
| Factor compatibility | negative agreement | neutral | positive agreement |

## From Binary to Ternary

**Before: binary factor graph message**

```rust
// Binary message: 2-element distribution
struct BinaryMsg {
    p_false: f64,
    p_true: f64,
}

// The margin of a binary variable sums to 1.0
// If p_false = p_true = 0.5, the variable is maximally uncertain
// But the graph has no way to say "this variable doesn't matter"
// The algorithm will keep oscillating, trying to pin it down
```

**After: ternary message**

```rust
// Ternary message: 3-element distribution
struct TernaryMsg {
    p_neg: f64,   // P(variable = -1)
    p_zero: f64,  // P(variable = 0) ← the "unknown" channel
    p_pos: f64,   // P(variable = +1)
}
```

The `p_zero` channel changes everything. When a variable has no evidence and no strong constraints from its neighbors, the belief converges to [0, 1, 0] — "I genuinely don't know" — instead of [0.5, 0, 0.5] which would suggest two equally likely but opposite interpretations.

**How loopy BP benefits:**

Binary BP on a frustrated graph (one where constraints conflict) oscillates. Each message flips between favoring true and false as the graph searches for a consistent assignment that may not exist. The iterations never converge.

Ternary BP on the same graph: the conflicting constraints push mass into the `p_zero` channel. The message "I think this variable should be +1, but the other factors say −1" gets resolved as "I don't know" instead of bouncing between extremes. The algorithm converges to a neutral belief instead of oscillating forever.

**0 is not nothing:** The `p_zero` mass isn't a cop-out — it's an informative signal. High `p_zero` means "the constraints on this variable are contradictory." In a factor graph for error correction, `p_zero` mass flags ambiguous bits that need retransmission. In a social network model, `p_zero` identifies nodes with divided loyalties. The neutral channel doesn't hide information; it surfaces structural uncertainty.

```rust
// Before: BP on a binary graph with frustrated constraints
// for i in 0..1000 { bp_step(&mut graph) }  // still oscillating

// After: BP on a ternary graph with frustrated constraints
// for i in 0..5 { bp_step(&mut graph) }  // converged to neutral beliefs
```

## Why It Matters

Ternary belief propagation doesn't fight ambiguity — it preserves it. The `0` channel carries "I don't know" as a genuine belief state, preventing oscillations and enabling convergence on graphs that binary BP can't resolve. Messages are only 50% larger (3 floats vs 2), but they express a fundamentally richer vocabulary of uncertainty.
