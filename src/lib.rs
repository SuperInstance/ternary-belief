//! Belief propagation over factor graphs with ternary variables {-1, 0, +1}.

use std::collections::HashMap;

pub type TernaryVal = i8;
pub const TERNARY_VALS: [TernaryVal; 3] = [-1, 0, 1];

/// Probability distribution over {-1, 0, +1}.
pub type Message = [f64; 3];
pub type Marginal = [f64; 3];

pub fn ternary_idx(v: TernaryVal) -> usize {
    (v + 1) as usize
}

pub fn uniform_msg() -> Message {
    [1.0 / 3.0; 3]
}

pub fn normalize_msg(m: &mut Message) {
    let sum: f64 = m.iter().sum();
    if sum > 1e-12 {
        for x in m.iter_mut() {
            *x /= sum;
        }
    }
}

pub fn point_mass(v: TernaryVal) -> Message {
    let mut m = [0.0; 3];
    m[ternary_idx(v)] = 1.0;
    m
}

/// A variable node in the factor graph — takes values in {-1, 0, +1}.
#[derive(Debug, Clone)]
pub struct Variable {
    pub id: usize,
    pub name: String,
    pub evidence: Option<TernaryVal>,
}

/// A factor node encoding a joint probability table over its variables.
#[derive(Debug, Clone)]
pub struct Factor {
    pub id: usize,
    pub var_ids: Vec<usize>,
    /// Flat probability table, row-major over var_ids, each dim size 3.
    pub table: Vec<f64>,
}

impl Factor {
    pub fn new(id: usize, var_ids: Vec<usize>, table: Vec<f64>) -> Self {
        let expected = 3_usize.pow(var_ids.len() as u32);
        assert_eq!(
            table.len(),
            expected,
            "table must have {} entries for {} vars",
            expected,
            var_ids.len()
        );
        Self { id, var_ids, table }
    }

    /// Get probability for a flat assignment (each element is 0,1,2 index).
    pub fn get_prob(&self, assignment: &[usize]) -> f64 {
        let mut idx = 0;
        for &a in assignment {
            idx = idx * 3 + a;
        }
        self.table[idx]
    }

    /// Build a uniform factor over `n_vars` ternary variables.
    pub fn uniform(id: usize, var_ids: Vec<usize>) -> Self {
        let n = 3_usize.pow(var_ids.len() as u32);
        Self::new(id, var_ids, vec![1.0 / n as f64; n])
    }
}

/// Bipartite factor graph: variable nodes ↔ factor nodes.
pub struct FactorGraph {
    pub variables: Vec<Variable>,
    pub factors: Vec<Factor>,
    pub var_to_factors: HashMap<usize, Vec<usize>>,
}

impl FactorGraph {
    pub fn new() -> Self {
        Self {
            variables: Vec::new(),
            factors: Vec::new(),
            var_to_factors: HashMap::new(),
        }
    }

    pub fn add_variable(&mut self, name: &str) -> usize {
        let id = self.variables.len();
        self.variables.push(Variable {
            id,
            name: name.to_string(),
            evidence: None,
        });
        self.var_to_factors.insert(id, Vec::new());
        id
    }

    pub fn add_factor(&mut self, var_ids: Vec<usize>, table: Vec<f64>) -> usize {
        let id = self.factors.len();
        for &v in &var_ids {
            self.var_to_factors.entry(v).or_default().push(id);
        }
        self.factors.push(Factor::new(id, var_ids, table));
        id
    }

    /// Clamp a variable to an observed value.
    pub fn clamp_evidence(&mut self, var_id: usize, value: TernaryVal) {
        if let Some(v) = self.variables.get_mut(var_id) {
            v.evidence = Some(value);
        }
    }

    /// Remove evidence from a variable.
    pub fn clear_evidence(&mut self, var_id: usize) {
        if let Some(v) = self.variables.get_mut(var_id) {
            v.evidence = None;
        }
    }

    pub fn n_vars(&self) -> usize {
        self.variables.len()
    }

    pub fn n_factors(&self) -> usize {
        self.factors.len()
    }
}

impl Default for FactorGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Sum-product message passing.
pub struct MessagePassing {
    /// μ_{x→f}: variable x to factor f
    pub var_to_factor: HashMap<(usize, usize), Message>,
    /// μ_{f→x}: factor f to variable x
    pub factor_to_var: HashMap<(usize, usize), Message>,
}

impl MessagePassing {
    pub fn new(graph: &FactorGraph) -> Self {
        let mut var_to_factor = HashMap::new();
        let mut factor_to_var = HashMap::new();
        for f in &graph.factors {
            for &v in &f.var_ids {
                var_to_factor.insert((v, f.id), uniform_msg());
                factor_to_var.insert((f.id, v), uniform_msg());
            }
        }
        Self {
            var_to_factor,
            factor_to_var,
        }
    }

    /// μ_{x→f}(x) = evidence (hard) or product of all other incoming factor messages.
    pub fn update_var_to_factor(
        &mut self,
        graph: &FactorGraph,
        var_id: usize,
        factor_id: usize,
    ) {
        let var = &graph.variables[var_id];
        if let Some(ev) = var.evidence {
            self.var_to_factor.insert((var_id, factor_id), point_mass(ev));
            return;
        }

        let mut msg = uniform_msg();
        let flist = graph.var_to_factors.get(&var_id).cloned().unwrap_or_default();
        for g_id in flist {
            if g_id == factor_id {
                continue;
            }
            if let Some(&in_msg) = self.factor_to_var.get(&(g_id, var_id)) {
                for i in 0..3 {
                    msg[i] *= in_msg[i];
                }
            }
        }
        normalize_msg(&mut msg);
        self.var_to_factor.insert((var_id, factor_id), msg);
    }

    /// μ_{f→x}(x) = Σ_{~x} f(x,~x) ∏_{y≠x} μ_{y→f}(y).
    pub fn update_factor_to_var(
        &mut self,
        graph: &FactorGraph,
        factor_id: usize,
        var_id: usize,
    ) {
        let factor = &graph.factors[factor_id];
        let var_pos = factor.var_ids.iter().position(|&v| v == var_id).unwrap();
        let n_vars = factor.var_ids.len();
        let n_configs = 3_usize.pow(n_vars as u32);

        let mut msg = [0.0f64; 3];

        for config in 0..n_configs {
            // decode assignment
            let mut assignment = vec![0usize; n_vars];
            let mut tmp = config;
            for i in (0..n_vars).rev() {
                assignment[i] = tmp % 3;
                tmp /= 3;
            }

            let fp = factor.get_prob(&assignment);
            if fp < 1e-15 {
                continue;
            }

            // product of incoming messages from all *other* variables
            let mut prod = fp;
            for (j, &other_var_id) in factor.var_ids.iter().enumerate() {
                if j == var_pos {
                    continue;
                }
                if let Some(&in_msg) = self.var_to_factor.get(&(other_var_id, factor_id)) {
                    prod *= in_msg[assignment[j]];
                }
            }

            msg[assignment[var_pos]] += prod;
        }

        normalize_msg(&mut msg);
        self.factor_to_var.insert((factor_id, var_id), msg);
    }

    /// Marginal P(x) ∝ ∏_f μ_{f→x}(x).
    pub fn marginal(&self, graph: &FactorGraph, var_id: usize) -> Marginal {
        let var = &graph.variables[var_id];
        if let Some(ev) = var.evidence {
            return point_mass(ev);
        }

        let mut m = uniform_msg();
        let flist = graph.var_to_factors.get(&var_id).cloned().unwrap_or_default();
        for f_id in flist {
            if let Some(&in_msg) = self.factor_to_var.get(&(f_id, var_id)) {
                for i in 0..3 {
                    m[i] *= in_msg[i];
                }
            }
        }
        normalize_msg(&mut m);
        m
    }

    /// Run one full round: all factor→var then all var→factor updates.
    pub fn run_round(&mut self, graph: &FactorGraph) {
        for f in &graph.factors {
            for &v in &f.var_ids {
                self.update_factor_to_var(graph, f.id, v);
            }
        }
        for f in &graph.factors {
            for &v in &f.var_ids {
                self.update_var_to_factor(graph, v, f.id);
            }
        }
    }
}

/// Loopy Belief Propagation — iterates message passing until convergence.
pub struct LoopyBeliefPropagation {
    pub max_iterations: usize,
    pub tolerance: f64,
}

impl LoopyBeliefPropagation {
    pub fn new(max_iterations: usize, tolerance: f64) -> Self {
        Self {
            max_iterations,
            tolerance,
        }
    }

    /// Returns (messages, converged).
    pub fn run(&self, graph: &FactorGraph) -> (MessagePassing, bool) {
        let mut msgs = MessagePassing::new(graph);

        for _iter in 0..self.max_iterations {
            let old_vtf = msgs.var_to_factor.clone();

            msgs.run_round(graph);

            // Check convergence via max change in var→factor messages
            let delta: f64 = msgs
                .var_to_factor
                .iter()
                .map(|(key, &new_m)| {
                    if let Some(&old_m) = old_vtf.get(key) {
                        new_m.iter()
                            .zip(old_m.iter())
                            .map(|(&n, &o)| (n - o).abs())
                            .sum::<f64>()
                    } else {
                        1.0
                    }
                })
                .sum();

            if delta < self.tolerance {
                return (msgs, true);
            }
        }

        (msgs, false)
    }

    /// Compute all marginals after convergence.
    pub fn marginals(&self, graph: &FactorGraph) -> Vec<Marginal> {
        let (msgs, _) = self.run(graph);
        (0..graph.n_vars())
            .map(|v| msgs.marginal(graph, v))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_chain() -> FactorGraph {
        // x0 — f01 — x1
        // f01 table: symmetric preference for same value
        let mut g = FactorGraph::new();
        let x0 = g.add_variable("x0");
        let x1 = g.add_variable("x1");
        // table[i*3+j] = P(x0=i, x1=j), prefer matching
        let mut table = vec![0.1; 9];
        table[0] = 0.5; // (-1,-1)
        table[4] = 0.5; // (0,0)
        table[8] = 0.5; // (+1,+1)
        let sum: f64 = table.iter().sum();
        let table: Vec<f64> = table.iter().map(|&t| t / sum).collect();
        g.add_factor(vec![x0, x1], table);
        g
    }

    #[test]
    fn test_add_variable_increments_id() {
        let mut g = FactorGraph::new();
        let id0 = g.add_variable("a");
        let id1 = g.add_variable("b");
        assert_eq!(id0, 0);
        assert_eq!(id1, 1);
    }

    #[test]
    fn test_add_factor_links_vars() {
        let g = simple_chain();
        assert_eq!(g.n_factors(), 1);
        let flist0 = g.var_to_factors.get(&0).unwrap();
        let flist1 = g.var_to_factors.get(&1).unwrap();
        assert!(flist0.contains(&0));
        assert!(flist1.contains(&0));
    }

    #[test]
    fn test_factor_get_prob() {
        let f = Factor::new(0, vec![0, 1], vec![0.1; 9]);
        let p = f.get_prob(&[0, 0]);
        assert!((p - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_factor_uniform_has_equal_probs() {
        let f = Factor::uniform(0, vec![0, 1]);
        let p00 = f.get_prob(&[0, 0]);
        let p11 = f.get_prob(&[1, 1]);
        assert!((p00 - p11).abs() < 1e-10);
    }

    #[test]
    fn test_clamp_evidence_sets_point_mass() {
        let mut g = simple_chain();
        g.clamp_evidence(0, -1);
        let mut msgs = MessagePassing::new(&g);
        msgs.update_var_to_factor(&g, 0, 0);
        let m = msgs.var_to_factor[&(0, 0)];
        assert!((m[0] - 1.0).abs() < 1e-10);
        assert!(m[1].abs() < 1e-10);
        assert!(m[2].abs() < 1e-10);
    }

    #[test]
    fn test_clear_evidence_restores_uniform() {
        let mut g = simple_chain();
        g.clamp_evidence(0, 1);
        g.clear_evidence(0);
        assert!(g.variables[0].evidence.is_none());
    }

    #[test]
    fn test_unary_factor_marginal() {
        // single variable x0 with unary factor [0.6, 0.3, 0.1]
        let mut g = FactorGraph::new();
        let x0 = g.add_variable("x0");
        g.add_factor(vec![x0], vec![0.6, 0.3, 0.1]);
        let (msgs, _) = LoopyBeliefPropagation::new(50, 1e-6).run(&g);
        let m = msgs.marginal(&g, x0);
        assert!(m[0] > m[1], "mode should be at -1");
        let sum: f64 = m.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_lbp_chain_marginals_sum_to_one() {
        let g = simple_chain();
        let lbp = LoopyBeliefPropagation::new(100, 1e-6);
        let marginals = lbp.marginals(&g);
        for m in &marginals {
            let sum: f64 = m.iter().sum();
            assert!((sum - 1.0).abs() < 1e-9, "marginal sums to {sum}");
        }
    }

    #[test]
    fn test_lbp_converges_on_tree() {
        // Tree ⇒ exact BP, must converge
        let g = simple_chain();
        let lbp = LoopyBeliefPropagation::new(100, 1e-8);
        let (_, converged) = lbp.run(&g);
        assert!(converged, "BP should converge on a tree");
    }

    #[test]
    fn test_evidence_propagates_to_neighbor() {
        // x0 — f — x1: clamp x0=-1, so x1 should also prefer -1
        let g = {
            let mut g = FactorGraph::new();
            let x0 = g.add_variable("x0");
            let x1 = g.add_variable("x1");
            // strongly prefer matching
            let mut table = vec![0.01; 9];
            table[0] = 0.9; // (-1,-1)
            table[4] = 0.9; // (0,0)
            table[8] = 0.9; // (+1,+1)
            let s: f64 = table.iter().sum();
            let t: Vec<f64> = table.iter().map(|&v| v / s).collect();
            g.add_factor(vec![x0, x1], t);
            g
        };
        let mut g = g;
        g.clamp_evidence(0, -1);
        let lbp = LoopyBeliefPropagation::new(100, 1e-8);
        let marginals = lbp.marginals(&g);
        // x1 should mostly be -1
        assert!(
            marginals[1][0] > marginals[1][1],
            "x1 should prefer -1 when x0=-1"
        );
    }

    #[test]
    fn test_marginal_with_evidence_is_point_mass() {
        let mut g = simple_chain();
        g.clamp_evidence(0, 0); // x0 = 0
        let mut msgs = MessagePassing::new(&g);
        msgs.run_round(&g);
        let m = msgs.marginal(&g, 0);
        assert!((m[1] - 1.0).abs() < 1e-10); // index 1 = value 0
    }

    #[test]
    fn test_point_mass_helper() {
        let m = point_mass(1);
        assert!((m[2] - 1.0).abs() < 1e-10);
        assert!(m[0].abs() < 1e-10);
        assert!(m[1].abs() < 1e-10);
    }

    #[test]
    fn test_ternary_idx() {
        assert_eq!(ternary_idx(-1), 0);
        assert_eq!(ternary_idx(0), 1);
        assert_eq!(ternary_idx(1), 2);
    }
}
