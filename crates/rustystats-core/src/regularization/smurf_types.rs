use ndarray::Array2;

#[derive(Debug, Clone, PartialEq)]
pub enum SmurfPenaltyType {
    None,
    Lasso,
    GroupLasso { group_id: usize },
    FusedLasso,
    GenFusedLasso,
    GraphGuidedFusedLasso,
    TwoDFusedLasso { n_rows: usize, n_cols: usize },
}

impl SmurfPenaltyType {
    pub fn needs_adj_matrix(&self) -> bool {
        matches!(self, SmurfPenaltyType::GraphGuidedFusedLasso)
    }
    pub fn is_none(&self) -> bool {
        matches!(self, SmurfPenaltyType::None)
    }
    pub fn uses_admm(&self) -> bool {
        matches!(self,
            SmurfPenaltyType::FusedLasso
            | SmurfPenaltyType::GenFusedLasso
            | SmurfPenaltyType::GraphGuidedFusedLasso
            | SmurfPenaltyType::TwoDFusedLasso { .. }
        )
    }
    pub fn name(&self) -> &'static str {
        match self {
            SmurfPenaltyType::None => "none",
            SmurfPenaltyType::Lasso => "lasso",
            SmurfPenaltyType::GroupLasso { .. } => "grouplasso",
            SmurfPenaltyType::FusedLasso => "flasso",
            SmurfPenaltyType::GenFusedLasso => "gflasso",
            SmurfPenaltyType::GraphGuidedFusedLasso => "ggflasso",
            SmurfPenaltyType::TwoDFusedLasso { .. } => "2dflasso",
        }
    }
}

#[derive(Debug, Clone)]
pub struct SmurfTermSpec {
    pub name: String,
    pub penalty_type: SmurfPenaltyType,
    pub n_params: usize,
    pub ref_cat: Option<usize>,
    pub lambda1: f64,
    pub lambda2: f64,
    pub adj_matrix: Option<Array2<f64>>,
}

impl SmurfTermSpec {
    pub fn none(name: impl Into<String>, n_params: usize) -> Self {
        Self { name: name.into(), penalty_type: SmurfPenaltyType::None, n_params, ref_cat: None, lambda1: 0.0, lambda2: 0.0, adj_matrix: None }
    }
    pub fn lasso(name: impl Into<String>, n_params: usize) -> Self {
        Self { name: name.into(), penalty_type: SmurfPenaltyType::Lasso, n_params, ref_cat: None, lambda1: 0.0, lambda2: 0.0, adj_matrix: None }
    }
    pub fn fused_lasso(name: impl Into<String>, n_params: usize, ref_cat: Option<usize>) -> Self {
        Self { name: name.into(), penalty_type: SmurfPenaltyType::FusedLasso, n_params, ref_cat, lambda1: 0.0, lambda2: 0.0, adj_matrix: None }
    }
    pub fn graph_guided(name: impl Into<String>, n_params: usize, ref_cat: Option<usize>, adj_matrix: Array2<f64>) -> Self {
        Self { name: name.into(), penalty_type: SmurfPenaltyType::GraphGuidedFusedLasso, n_params, ref_cat, lambda1: 0.0, lambda2: 0.0, adj_matrix: Some(adj_matrix) }
    }
    pub fn group_lasso(name: impl Into<String>, n_params: usize, group_id: usize) -> Self {
        Self { name: name.into(), penalty_type: SmurfPenaltyType::GroupLasso { group_id }, n_params, ref_cat: None, lambda1: 0.0, lambda2: 0.0, adj_matrix: None }
    }
    pub fn with_lambda1(mut self, lambda1: f64) -> Self { self.lambda1 = lambda1; self }
    pub fn with_lambda2(mut self, lambda2: f64) -> Self { self.lambda2 = lambda2; self }
}

#[derive(Debug, Clone, PartialEq)]
pub enum LambdaSelection {
    Fixed(f64),
    CvDev,
    CvOneSeDev,
    CvMse,
    IsAic,
    IsBic,
    IsGcv,
}

impl Default for LambdaSelection {
    fn default() -> Self { LambdaSelection::CvOneSeDev }
}

impl LambdaSelection {
    pub fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "cv.dev"    => Ok(LambdaSelection::CvDev),
            "cv1se.dev" => Ok(LambdaSelection::CvOneSeDev),
            "cv.mse"    => Ok(LambdaSelection::CvMse),
            "is.aic"    => Ok(LambdaSelection::IsAic),
            "is.bic"    => Ok(LambdaSelection::IsBic),
            "is.gcv"    => Ok(LambdaSelection::IsGcv),
            other => other.parse::<f64>()
                .map(LambdaSelection::Fixed)
                .map_err(|_| format!("lambda_selection '{}' no reconocido. Opciones: cv.dev, cv1se.dev, cv.mse, is.aic, is.bic, is.gcv, o un número positivo.", other)),
        }
    }
    pub fn requires_grid_search(&self) -> bool {
        !matches!(self, LambdaSelection::Fixed(_))
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub enum PenWeightsStrategy {
    #[default]
    Eq,
    Stand,
    Glm,
    GlmStand,
}

impl PenWeightsStrategy {
    pub fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "eq"        => Ok(PenWeightsStrategy::Eq),
            "stand"     => Ok(PenWeightsStrategy::Stand),
            "glm"       => Ok(PenWeightsStrategy::Glm),
            "glm.stand" => Ok(PenWeightsStrategy::GlmStand),
            other => Err(format!("pen_weights '{}' no reconocido. Opciones: eq, stand, glm, glm.stand", other)),
        }
    }
    pub fn requires_glm_fit(&self) -> bool {
        matches!(self, PenWeightsStrategy::Glm | PenWeightsStrategy::GlmStand)
    }
}

#[derive(Debug, Clone)]
pub struct SmurfPenalty {
    pub terms: Vec<SmurfTermSpec>,
    pub lambda: f64,
    pub lambda_selection: LambdaSelection,
    pub cv_folds: usize,
    pub pen_weights: PenWeightsStrategy,
    pub reestimate: bool,
    pub epsilon: f64,
    pub max_iter: usize,
    pub step_init: f64,
    pub tau: f64,
    pub parallel_proximal: bool,
}

impl SmurfPenalty {
    pub fn new(terms: Vec<SmurfTermSpec>) -> Self {
        Self { terms, lambda: 1.0, lambda_selection: LambdaSelection::default(), cv_folds: 5, pen_weights: PenWeightsStrategy::default(), reestimate: true, epsilon: 1e-5, max_iter: 10_000, step_init: 1.0, tau: 0.5, parallel_proximal: false }
    }
    pub fn with_lambda(mut self, lambda: f64) -> Self { self.lambda = lambda; self.lambda_selection = LambdaSelection::Fixed(lambda); self }
    pub fn with_lambda_selection(mut self, sel: LambdaSelection) -> Self { self.lambda_selection = sel; self }
    pub fn with_cv_folds(mut self, k: usize) -> Self { self.cv_folds = k; self }
    pub fn with_pen_weights(mut self, strategy: PenWeightsStrategy) -> Self { self.pen_weights = strategy; self }
    pub fn without_reestimation(mut self) -> Self { self.reestimate = false; self }
    pub fn with_epsilon(mut self, epsilon: f64) -> Self { self.epsilon = epsilon; self }
    pub fn with_max_iter(mut self, n: usize) -> Self { self.max_iter = n; self }
    pub fn with_parallel_proximal(mut self) -> Self { self.parallel_proximal = true; self }
    pub fn with_step_init(mut self, step: f64) -> Self { self.step_init = step; self }
    pub fn with_tau(mut self, tau: f64) -> Self { self.tau = tau; self }

    pub fn total_params(&self) -> usize {
        self.terms.iter().map(|t| t.n_params).sum()
    }
    pub fn term_offsets(&self) -> Vec<usize> {
        let mut offsets = Vec::with_capacity(self.terms.len());
        let mut offset = 0usize;
        for term in &self.terms { offsets.push(offset); offset += term.n_params; }
        offsets
    }
    pub fn n_admm_terms(&self) -> usize {
        self.terms.iter().filter(|t| t.penalty_type.uses_admm()).count()
    }

    pub fn validate(&self) -> Result<(), String> {
        if let LambdaSelection::Fixed(lambda) = self.lambda_selection {
            if lambda <= 0.0 { return Err(format!("lambda debe ser positivo, recibido: {}", lambda)); }
        }
        if self.lambda_selection.requires_grid_search() && self.cv_folds < 2 {
            return Err(format!("cv_folds debe ser >= 2, recibido: {}", self.cv_folds));
        }
        if self.epsilon <= 0.0 { return Err(format!("epsilon debe ser positivo, recibido: {}", self.epsilon)); }
        if !(0.0 < self.tau && self.tau < 1.0) { return Err(format!("tau debe estar en (0,1), recibido: {}", self.tau)); }
        for (i, term) in self.terms.iter().enumerate() {
            if term.n_params == 0 { return Err(format!("Término '{}' (índice {}) tiene n_params = 0", term.name, i)); }
            if term.penalty_type.needs_adj_matrix() {
                let adj = term.adj_matrix.as_ref().ok_or_else(|| format!("Término '{}' necesita adj_matrix para ggflasso", term.name))?;
                Self::validate_adj_matrix(adj, &term.name)?;
            } else if term.adj_matrix.is_some() {
                return Err(format!("Término '{}' tiene adj_matrix pero penalización '{}' no la usa", term.name, term.penalty_type.name()));
            }
            if term.lambda1 < 0.0 { return Err(format!("lambda1 para '{}' debe ser >= 0", term.name)); }
            if term.lambda2 < 0.0 { return Err(format!("lambda2 para '{}' debe ser >= 0", term.name)); }
        }
        Ok(())
    }

    pub fn validate_adj_matrix(adj: &Array2<f64>, name: &str) -> Result<(), String> {
        let (rows, cols) = adj.dim();
        if rows != cols { return Err(format!("adj_matrix para '{}' debe ser cuadrada, recibida: {}x{}", name, rows, cols)); }
        let tol = 1e-10;
        for i in 0..rows {
            for j in 0..cols {
                if (adj[[i,j]] - adj[[j,i]]).abs() > tol { return Err(format!("adj_matrix para '{}' no es simétrica en [{},{}]", name, i, j)); }
            }
        }
        for &v in adj.iter() {
            if (v - 0.0).abs() > tol && (v - 1.0).abs() > tol { return Err(format!("adj_matrix para '{}' solo puede contener 0 o 1, encontrado: {}", name, v)); }
        }
        for i in 0..rows {
            if adj[[i,i]] > tol { return Err(format!("adj_matrix para '{}' tiene auto-arista en diagonal [{}]", name, i)); }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_penalty_type_names() {
        assert_eq!(SmurfPenaltyType::None.name(), "none");
        assert_eq!(SmurfPenaltyType::Lasso.name(), "lasso");
        assert_eq!(SmurfPenaltyType::FusedLasso.name(), "flasso");
        assert_eq!(SmurfPenaltyType::GraphGuidedFusedLasso.name(), "ggflasso");
    }

    #[test]
    fn test_penalty_type_uses_admm() {
        assert!(!SmurfPenaltyType::None.uses_admm());
        assert!(!SmurfPenaltyType::Lasso.uses_admm());
        assert!(!SmurfPenaltyType::GroupLasso { group_id: 0 }.uses_admm());
        assert!(SmurfPenaltyType::FusedLasso.uses_admm());
        assert!(SmurfPenaltyType::GenFusedLasso.uses_admm());
        assert!(SmurfPenaltyType::GraphGuidedFusedLasso.uses_admm());
        assert!(SmurfPenaltyType::TwoDFusedLasso { n_rows: 3, n_cols: 4 }.uses_admm());
    }

    #[test]
    fn test_lambda_selection_from_str() {
        assert_eq!(LambdaSelection::from_str("cv.dev").unwrap(), LambdaSelection::CvDev);
        assert_eq!(LambdaSelection::from_str("is.aic").unwrap(), LambdaSelection::IsAic);
        assert_eq!(LambdaSelection::from_str("0.05").unwrap(), LambdaSelection::Fixed(0.05));
        assert!(LambdaSelection::from_str("invalid").is_err());
    }

    #[test]
    fn test_pen_weights_from_str() {
        assert_eq!(PenWeightsStrategy::from_str("eq").unwrap(), PenWeightsStrategy::Eq);
        assert_eq!(PenWeightsStrategy::from_str("glm.stand").unwrap(), PenWeightsStrategy::GlmStand);
        assert!(PenWeightsStrategy::from_str("gam").is_err());
    }

    #[test]
    fn test_smurf_penalty_constructors() {
        let terms = vec![
            SmurfTermSpec::lasso("bonus_malus", 1),
            SmurfTermSpec::fused_lasso("driver_age", 4, Some(0)),
            SmurfTermSpec::group_lasso("brand", 10, 0),
        ];
        let penalty = SmurfPenalty::new(terms).with_lambda_selection(LambdaSelection::CvDev).with_cv_folds(5);
        assert_eq!(penalty.total_params(), 15);
        assert_eq!(penalty.term_offsets(), vec![0, 1, 5]);
        assert_eq!(penalty.n_admm_terms(), 1);
    }

    #[test]
    fn test_validate_ok() {
        let penalty = SmurfPenalty::new(vec![SmurfTermSpec::lasso("x", 3)]).with_lambda(0.1);
        assert!(penalty.validate().is_ok());
    }

    #[test]
    fn test_validate_lambda_zero() {
        let penalty = SmurfPenalty::new(vec![SmurfTermSpec::lasso("x", 1)]).with_lambda(0.0);
        assert!(penalty.validate().is_err());
    }

    #[test]
    fn test_validate_cv_folds_too_small() {
        let mut penalty = SmurfPenalty::new(vec![SmurfTermSpec::lasso("x", 1)]);
        penalty.cv_folds = 1;
        assert!(penalty.validate().is_err());
    }

    #[test]
    fn test_validate_missing_adj_matrix() {
        let term = SmurfTermSpec { name: "region".into(), penalty_type: SmurfPenaltyType::GraphGuidedFusedLasso, n_params: 5, ref_cat: None, lambda1: 0.0, lambda2: 0.0, adj_matrix: None };
        assert!(SmurfPenalty::new(vec![term]).validate().is_err());
    }

    #[test]
    fn test_adj_matrix_valid() {
        let adj = array![[0.0,1.0,0.0],[1.0,0.0,1.0],[0.0,1.0,0.0]];
        let term = SmurfTermSpec::graph_guided("region", 2, Some(0), adj);
        assert!(SmurfPenalty::new(vec![term]).validate().is_ok());
    }

    #[test]
    fn test_adj_matrix_not_symmetric() {
        let adj = array![[0.0,1.0],[0.0,0.0]];
        assert!(SmurfPenalty::validate_adj_matrix(&adj, "test").is_err());
    }

    #[test]
    fn test_adj_matrix_non_binary() {
        let adj = array![[0.0,0.5],[0.5,0.0]];
        assert!(SmurfPenalty::validate_adj_matrix(&adj, "test").is_err());
    }

    #[test]
    fn test_adj_matrix_self_loop() {
        let adj = array![[1.0,0.0],[0.0,0.0]];
        assert!(SmurfPenalty::validate_adj_matrix(&adj, "test").is_err());
    }

    #[test]
    fn test_term_offsets() {
        let terms = vec![SmurfTermSpec::lasso("a", 1), SmurfTermSpec::fused_lasso("b", 5, Some(0)), SmurfTermSpec::group_lasso("c", 3, 0)];
        let penalty = SmurfPenalty::new(terms);
        assert_eq!(penalty.term_offsets(), vec![0, 1, 6]);
        assert_eq!(penalty.total_params(), 9);
    }
}
