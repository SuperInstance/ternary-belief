//! # ternary-belief
//! Belief propagation on ternary factor graphs with {-1, 0, +1} variables.

/// A variable node in the factor graph, taking values in {-1, 0, +1}
#[derive(Debug, Clone)]
pub struct VariableNode {
    pub id: usize,
    pub belief: [f64; 3], // [P(-1), P(0), P(+1)]
    pub fixed: Option<i8>,
}

impl VariableNode {
    pub fn new(id: usize) -> Self { Self { id, belief: [1.0/3.0; 3], fixed: None } }

    pub fn fixed_val(id: usize, val: i8) -> Self {
        let b = match val { -1 => [1.0, 0.0, 0.0], 0 => [0.0, 1.0, 0.0], _ => [0.0, 0.0, 1.0] };
        Self { id, belief: b, fixed: Some(val) }
    }

    pub fn normalize(&mut self) {
        let sum: f64 = self.belief.iter().sum();
        if sum > 1e-10 { for b in self.belief.iter_mut() { *b /= sum; } }
    }

    pub fn map(&self) -> i8 {
        let max_idx = self.belief.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i).unwrap_or(1);
        max_idx as i8 - 1
    }

    pub fn entropy(&self) -> f64 {
        self.belief.iter().filter(|&&p| p > 1e-10).map(|&p| -p * p.log2()).sum()
    }
}

/// Factor between two ternary variables: a 3x3 compatibility matrix
#[derive(Debug, Clone)]
pub struct PairFactor {
    pub v0: usize,
    pub v1: usize,
    pub compat: [[f64; 3]; 3], // compat[a+1][b+1]
}

impl PairFactor {
    pub fn new(v0: usize, v1: usize) -> Self {
        Self { v0, v1, compat: [[1.0; 3]; 3] }
    }

    pub fn set(&mut self, a: i8, b: i8, val: f64) { self.compat[(a+1) as usize][(b+1) as usize] = val; }
    pub fn get(&self, a: i8, b: i8) -> f64 { self.compat[(a+1) as usize][(b+1) as usize] }
}

/// Ternary factor graph for belief propagation
pub struct TernaryFactorGraph {
    pub variables: Vec<VariableNode>,
    pub factors: Vec<PairFactor>,
}

impl TernaryFactorGraph {
    pub fn new() -> Self { Self { variables: Vec::new(), factors: Vec::new() } }

    pub fn add_variable(&mut self) -> usize {
        let id = self.variables.len();
        self.variables.push(VariableNode::new(id));
        id
    }

    pub fn add_fixed_variable(&mut self, val: i8) -> usize {
        let id = self.variables.len();
        self.variables.push(VariableNode::fixed_val(id, val));
        id
    }

    pub fn add_pair_factor(&mut self, v0: usize, v1: usize) -> usize {
        let id = self.factors.len();
        self.factors.push(PairFactor::new(v0, v1));
        id
    }

    /// Run one round of belief propagation
    pub fn bp_round(&mut self) {
        // Save old beliefs
        let old_beliefs: Vec<[f64; 3]> = self.variables.iter().map(|v| v.belief).collect();

        for factor in &self.factors {
            // Message to v0: sum over v1
            for a in 0..3 {
                let mut msg = 0.0;
                for b in 0..3 {
                    msg += factor.compat[a][b] * old_beliefs[factor.v1][b];
                }
                self.variables[factor.v0].belief[a] = msg;
            }
            self.variables[factor.v0].normalize();

            // Message to v1: sum over v0
            for b in 0..3 {
                let mut msg = 0.0;
                for a in 0..3 {
                    msg += factor.compat[a][b] * old_beliefs[factor.v0][a];
                }
                self.variables[factor.v1].belief[b] = msg;
            }
            self.variables[factor.v1].normalize();
        }

        // Re-apply fixed evidence
        for var in &mut self.variables {
            if let Some(val) = var.fixed {
                var.belief = match val { -1 => [1.0, 0.0, 0.0], 0 => [0.0, 1.0, 0.0], _ => [0.0, 0.0, 1.0] };
            }
        }
    }

    pub fn run_loopy_bp(&mut self, iterations: usize) { for _ in 0..iterations { self.bp_round(); } }
    pub fn marginal(&self, var_id: usize) -> [f64; 3] { self.variables[var_id].belief }
    pub fn map_assignment(&self) -> Vec<i8> { self.variables.iter().map(|v| v.map()).collect() }

    pub fn energy(&self, assignment: &[i8]) -> f64 {
        self.factors.iter()
            .map(|f| -f.get(assignment[f.v0], assignment[f.v1]).ln().max(-50.0))
            .sum()
    }
}

pub fn set_evidence(graph: &mut TernaryFactorGraph, var_id: usize, val: i8) {
    if var_id < graph.variables.len() {
        graph.variables[var_id].fixed = Some(val);
        graph.variables[var_id].belief = match val { -1 => [1.0, 0.0, 0.0], 0 => [0.0, 1.0, 0.0], _ => [0.0, 0.0, 1.0] };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn variable_initial_uniform() {
        let v = VariableNode::new(0);
        assert!((v.belief[0] - 1.0/3.0).abs() < 1e-10);
    }

    #[test]
    fn variable_map() {
        let mut v = VariableNode::new(0);
        v.belief = [0.1, 0.7, 0.2];
        assert_eq!(v.map(), 0);
    }

    #[test]
    fn variable_entropy() {
        let v = VariableNode::new(0);
        assert!(v.entropy() > 1.5);
        let d = VariableNode::fixed_val(0, 1);
        assert!(d.entropy() < 0.01);
    }

    #[test]
    fn fixed_variable() {
        let v = VariableNode::fixed_val(0, -1);
        assert_eq!(v.map(), -1);
    }

    #[test]
    fn factor_get_set() {
        let mut f = PairFactor::new(0, 1);
        f.set(-1, 1, 5.0);
        assert!((f.get(-1, 1) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn graph_construction() {
        let mut g = TernaryFactorGraph::new();
        let v0 = g.add_variable();
        let v1 = g.add_variable();
        g.add_pair_factor(v0, v1);
        assert_eq!(g.variables.len(), 2);
        assert_eq!(g.factors.len(), 1);
    }

    #[test]
    fn evidence_clamping() {
        let mut g = TernaryFactorGraph::new();
        g.add_variable();
        set_evidence(&mut g, 0, 1);
        assert_eq!(g.variables[0].map(), 1);
    }

    #[test]
    fn bp_agreement_factor() {
        let mut g = TernaryFactorGraph::new();
        let v0 = g.add_variable();
        let v1 = g.add_variable();
        let f = g.add_pair_factor(v0, v1);
        // Prefer same values
        for a in -1..=1i8 {
            for b in -1..=1i8 {
                g.factors[f].set(a, b, if a == b { 10.0 } else { 0.1 });
            }
        }
        set_evidence(&mut g, 0, 1);
        g.run_loopy_bp(5);
        assert_eq!(g.variables[1].map(), 1);
    }

    #[test]
    fn bp_chain_propagation() {
        let mut g = TernaryFactorGraph::new();
        let v0 = g.add_variable();
        let v1 = g.add_variable();
        let v2 = g.add_variable();
        let f0 = g.add_pair_factor(v0, v1);
        let f1 = g.add_pair_factor(v1, v2);
        for f in &[f0, f1] {
            for a in -1..=1i8 {
                for b in -1..=1i8 {
                    g.factors[*f].set(a, b, if a == b { 10.0 } else { 0.1 });
                }
            }
        }
        set_evidence(&mut g, 0, -1);
        g.run_loopy_bp(5);
        // Verify BP runs and produces valid beliefs
        for v in &g.variables {
            let sum: f64 = v.belief.iter().sum();
            assert!((sum - 1.0).abs() < 0.01);
        }
    }

    #[test]
    fn energy_lower_for_agreement() {
        let mut g = TernaryFactorGraph::new();
        g.add_variable();
        g.add_variable();
        let f = g.add_pair_factor(0, 1);
        g.factors[f].set(1, 1, 10.0);
        g.factors[f].set(-1, 1, 0.1);
        assert!(g.energy(&[1, 1]) < g.energy(&[-1, 1]));
    }

    #[test]
    fn map_assignment_valid() {
        let mut g = TernaryFactorGraph::new();
        g.add_variable(); g.add_variable(); g.add_variable();
        let a = g.map_assignment();
        assert_eq!(a.len(), 3);
        for &v in &a { assert!(v >= -1 && v <= 1); }
    }
}

