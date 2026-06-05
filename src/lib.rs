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
    pub fn new(id: usize) -> Self {
        Self { id, belief: [1.0/3.0; 3], fixed: None }
    }

    pub fn fixed(id: usize, val: i8) -> Self {
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
        self.belief.iter()
            .filter(|&&p| p > 1e-10)
            .map(|&p| -p * p.log2())
            .sum()
    }
}

/// A factor node connecting variables with a compatibility function
#[derive(Debug, Clone)]
pub struct FactorNode {
    pub id: usize,
    pub variables: Vec<usize>,
    pub table: Vec<f64>, // Compatibility values for each variable assignment combination
}

impl FactorNode {
    pub fn new(id: usize, variables: Vec<usize>, arity: usize) -> Self {
        Self { id, variables, table: vec![1.0; 3usize.pow(arity as u32)] }
    }

    pub fn set_compatibility(&mut self, indices: &[usize], value: f64) {
        let idx = indices.iter().enumerate()
            .map(|(d, &v)| (v + 1) as usize * 3usize.pow((indices.len() - 1 - d) as u32))
            .sum::<usize>();
        if idx < self.table.len() { self.table[idx] = value; }
    }

    pub fn get_compatibility(&self, assignment: &[i8]) -> f64 {
        let idx = assignment.iter().enumerate()
            .map(|(d, &v)| (v + 1) as usize * 3usize.pow((assignment.len() - 1 - d) as u32))
            .sum::<usize>();
        self.table.get(idx).copied().unwrap_or(0.0)
    }
}

/// Ternary factor graph for belief propagation
pub struct TernaryFactorGraph {
    pub variables: Vec<VariableNode>,
    pub factors: Vec<FactorNode>,
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
        self.variables.push(VariableNode::fixed(id, val));
        id
    }

    pub fn add_factor(&mut self, variables: Vec<usize>) -> usize {
        let id = self.factors.len();
        let arity = variables.len();
        self.factors.push(FactorNode::new(id, variables, arity));
        id
    }

    /// Run one round of belief propagation (sum-product)
    pub fn bp_round(&mut self) {
        let n_vars = self.variables.len();
        let n_factors = self.factors.len();

        // Variable-to-factor messages (product of incoming factor messages)
        let mut var_to_factor = vec![[1.0; 3]; n_vars];
        for (vi, var) in self.variables.iter().enumerate() {
            var_to_factor[vi] = var.belief;
        }

        // Factor-to-variable messages
        for factor in &self.factors {
            if factor.variables.len() != 2 { continue; }
            let v0 = factor.variables[0];
            let v1 = factor.variables[1];

            // Message from factor to v0: sum over v1
            for a in 0..3 {
                let ai = a as i8 - 1;
                let mut msg = 0.0;
                for b in 0..3 {
                    let bi = b as i8 - 1;
                    let compat = factor.get_compatibility(&[ai, bi]);
                    msg += compat * var_to_factor[v1][b];
                }
                self.variables[v0].belief[a] = msg;
            }
            self.variables[v0].normalize();

            // Message from factor to v1: sum over v0
            for b in 0..3 {
                let bi = b as i8 - 1;
                let mut msg = 0.0;
                for a in 0..3 {
                    let ai = a as i8 - 1;
                    let compat = factor.get_compatibility(&[ai, bi]);
                    msg += compat * var_to_factor[v0][a];
                }
                self.variables[v1].belief[b] = msg;
            }
            self.variables[v1].normalize();
        }

        // Re-apply fixed evidence
        for var in &mut self.variables {
            if let Some(val) = var.fixed {
                var.belief = match val {
                    -1 => [1.0, 0.0, 0.0],
                    0 => [0.0, 1.0, 0.0],
                    _ => [0.0, 0.0, 1.0],
                };
            }
        }
    }

    /// Run loopy BP for N iterations
    pub fn run_loopy_bp(&mut self, iterations: usize) {
        for _ in 0..iterations {
            self.bp_round();
        }
    }

    /// Get marginal distribution for a variable
    pub fn marginal(&self, var_id: usize) -> [f64; 3] {
        self.variables[var_id].belief
    }

    /// Get MAP assignment for all variables
    pub fn map_assignment(&self) -> Vec<i8> {
        self.variables.iter().map(|v| v.map()).collect()
    }

    /// Total energy: -sum of log-compatibilities under current assignment
    pub fn energy(&self, assignment: &[i8]) -> f64 {
        self.factors.iter()
            .map(|f| {
                let vars: Vec<i8> = f.variables.iter()
                    .map(|&vi| assignment[vi])
                    .collect();
                -f.get_compatibility(&vars).ln().max(-50.0)
            })
            .sum()
    }
}

/// Evidence: clamp a variable to a specific value
pub fn set_evidence(graph: &mut TernaryFactorGraph, var_id: usize, val: i8) {
    if var_id < graph.variables.len() {
        graph.variables[var_id].fixed = Some(val);
        graph.variables[var_id].belief = match val {
            -1 => [1.0, 0.0, 0.0],
            0 => [0.0, 1.0, 0.0],
            _ => [0.0, 0.0, 1.0],
        };
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
        let d = VariableNode::fixed(0, 1);
        assert!(d.entropy() < 0.01);
    }

    #[test]
    fn fixed_variable() {
        let v = VariableNode::fixed(0, -1);
        assert_eq!(v.map(), -1);
        assert_eq!(v.fixed, Some(-1));
    }

    #[test]
    fn factor_compatibility() {
        let mut f = FactorNode::new(0, vec![0, 1], 2);
        f.set_compatibility(&[0, 2], 5.0); // -1 and +1
        assert!((f.get_compatibility(&[-1, 1]) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn graph_construction() {
        let mut g = TernaryFactorGraph::new();
        let v0 = g.add_variable();
        let v1 = g.add_variable();
        g.add_factor(vec![v0, v1]);
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
    fn bp_with_agreement() {
        // Two variables with factor preferring same values
        let mut g = TernaryFactorGraph::new();
        let v0 = g.add_variable();
        let v1 = g.add_variable();
        let f = g.add_factor(vec![v0, v1]);

        // Set same-value preference
        for a in -1..=1 {
            for b in -1..=1 {
                if a == b { g.factors[f].set_compatibility(&[(a+1) as usize, (b+1) as usize], 10.0); }
                else { g.factors[f].set_compatibility(&[(a+1) as usize, (b+1) as usize], 0.1); }
            }
        }

        // Clamp v0 to +1
        set_evidence(&mut g, 0, 1);
        g.run_loopy_bp(5);

        // v1 should also prefer +1
        assert_eq!(g.variables[1].map(), 1);
    }

    #[test]
    fn bp_with_evidence_propagation() {
        let mut g = TernaryFactorGraph::new();
        let v0 = g.add_variable();
        let v1 = g.add_variable();
        let v2 = g.add_variable();
        g.add_factor(vec![v0, v1]);
        g.add_factor(vec![v1, v2]);

        // All prefer agreement
        for f in &mut g.factors {
            for a in -1..=1 {
                for b in -1..=1 {
                    if a == b { f.set_compatibility(&[(a+1) as usize, (b+1) as usize], 10.0); }
                }
            }
        }

        set_evidence(&mut g, 0, -1);
        g.run_loopy_bp(10);
        assert_eq!(g.variables[2].map(), -1);
    }

    #[test]
    fn energy_calculation() {
        let mut g = TernaryFactorGraph::new();
        g.add_variable();
        g.add_variable();
        let f = g.add_factor(vec![0, 1]);
        g.factors[f].set_compatibility(&[2, 2], 10.0); // +1, +1

        let e_agree = g.energy(&[1, 1]);
        let e_disagree = g.energy(&[-1, 1]);
        assert!(e_agree < e_disagree); // Lower energy for agreeing
    }

    #[test]
    fn map_assignment_length() {
        let mut g = TernaryFactorGraph::new();
        g.add_variable();
        g.add_variable();
        g.add_variable();
        let assignment = g.map_assignment();
        assert_eq!(assignment.len(), 3);
        for &a in &assignment { assert!(a >= -1 && a <= 1); }
    }
}
