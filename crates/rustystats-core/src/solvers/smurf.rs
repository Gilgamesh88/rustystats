// SMuRF — Solver FISTA — Fases 5 y 6 pendientes
use crate::regularization::smurf_types::SmurfPenalty;
use crate::solvers::IRLSResult;

pub struct SmurfResult {
    pub irls_result: IRLSResult,
    pub selected_lambda: f64,
    pub penalty_type_names: Vec<String>,
    pub objective_history: Vec<f64>,
    pub fista_iterations: usize,
    pub fista_converged: bool,
}

pub struct SmurfConfig {
    pub penalty: SmurfPenalty,
}

impl SmurfConfig {
    pub fn new(penalty: SmurfPenalty) -> Self {
        Self { penalty }
    }
}

pub fn fit_smurf_glm(
    _x: &ndarray::Array2<f64>,
    _y: &ndarray::Array1<f64>,
    _weights: Option<&ndarray::Array1<f64>>,
    _offset: Option<&ndarray::Array1<f64>>,
    _family: Box<dyn crate::families::Family>,
    _link: Box<dyn crate::links::Link>,
    _config: SmurfConfig,
) -> crate::error::Result<SmurfResult> {
    Err(crate::error::RustyStatsError::MissingParameter(
        "SMuRF solver (FISTA) — pendiente Fase 6 del roadmap".to_string(),
    ))
}
