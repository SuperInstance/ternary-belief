# PLUG_AND_PLAY — Belief

> Belief propagation on ternary networks {-1, 0, +1}

## 🚀 Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
ternary-belief = { git = "https://github.com/SuperInstance/ternary-belief" }
```

Use in your code:

```rust
use ternary_belief::{BeliefNetwork, TernaryFactorGraph};

let mut net = BeliefNetwork::new();
net.add_factor(|a, b| (a == b) as i8);
let beliefs = net.infer();
```

## 📚 Available Documentation

| Document | Description |
|----------|-------------|
| `docs/FROM_BINARY.md` | Understanding ternary concepts as a binary programmer |
| `docs/MIGRATION.md` | Version migration guide |
| `docs/FUTURE-INTEGRATION.md` | Planned features and roadmap |

## 🔗 Integration

This crate is part of the [SuperInstance ternary fleet](https://github.com/SuperInstance). It uses the canonical `Ternary` type from `ternary-types` for cross-crate compatibility.

## 📄 License

MIT
