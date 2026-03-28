// =============================================================================
// GLM Solvers
// =============================================================================
//
// This module contains algorithms for fitting Generalized Linear Models.
// The main algorithm is IRLS (Iteratively Reweighted Least Squares).
//
// HOW GLM FITTING WORKS (High-Level Overview)
// -------------------------------------------
//
// We want to find coefficients β that best explain the relationship:
//
//     g(E[Y]) = Xβ
//
// where:
//   - Y is the response variable (what we're predicting)
//   - X is the design matrix (predictors/features)
//   - β is the coefficient vector (what we're solving for)
//   - g is the link function
//   - E[Y] = μ is the expected value of Y
//
// Unlike ordinary least squares, we can't solve this directly because:
//   1. The link function g() makes it non-linear
//   2. The variance depends on μ (heteroscedasticity)
//
// IRLS solves this by iteratively:
//   1. Linearizing the problem around current estimates
//   2. Solving a weighted least squares problem
//   3. Updating estimates and repeating until convergence
//
// =============================================================================

mod coordinate_descent;
pub mod gcv_optimizer;
mod irls;
pub mod nnls;
pub mod smooth_glm;
pub mod smurf;

pub use gcv_optimizer::{brent_minimize, MultiTermGCVOptimizer};
pub use irls::{
    compute_xtwx, compute_xtwx_xtwz, solve_weighted_least_squares_with_penalty_matrix,
    solve_wls_from_precomputed,
};
pub use irls::{fit_glm_unified, FitConfig, IRLSConfig, IRLSResult};
pub use smurf::{fit_smurf_glm, SmurfConfig, SmurfResult};
pub use nnls::{
    nnls, nnls_penalized, nnls_weighted, nnls_weighted_penalized, NNLSConfig, NNLSResult,
};
pub use smooth_glm::{
    fit_smooth_glm_full_matrix, Monotonicity, SmoothGLMConfig, SmoothGLMResult, SmoothTermData,
    SmoothTermSpec,
};

use ndarray::{Array1, ArrayView2};
use rayon::prelude::*;

use crate::constants::MAX_IRLS_WEIGHT;
use crate::error::{Result, RustyStatsError};
use crate::families::Family;
use crate::links::Link;

/// Safe initialization of μ that works for any family.
///
/// Used as fallback when `family.initialize_mu(y)` produces invalid values
/// (e.g., all zeros for Poisson). Computes a weighted average of each y_i
/// with the global mean, then clamps to the family's valid range.
pub(crate) fn initialize_mu_safe(y: &Array1<f64>, family: &dyn Family) -> Array1<f64> {
    let y_mean = y.mean().unwrap_or(1.0).max(0.01);
    let raw: Array1<f64> = y.mapv(|yi| (yi + y_mean) / 2.0);
    family.clamp_mu(&raw)
}

// =============================================================================
// Shared GLM Input Validation
// =============================================================================

/// Validated and prepared GLM inputs (offset and weights ready to use).
pub(crate) struct ValidatedInputs {
    pub offset: Array1<f64>,
    pub prior_weights: Array1<f64>,
}

/// Validate GLM inputs and prepare offset/weights.
///
/// Checks dimension compatibility of X, y, offset, and weights.
/// Returns owned, ready-to-use offset and prior_weights arrays.
pub(crate) fn validate_glm_inputs(
    y: &Array1<f64>,
    x: ArrayView2<'_, f64>,
    offset: Option<&Array1<f64>>,
    weights: Option<&Array1<f64>>,
) -> Result<ValidatedInputs> {
    let n = y.len();
    let p = x.ncols();

    if x.nrows() != n {
        return Err(RustyStatsError::dim_mismatch(
            n,
            x.nrows(),
            "X rows vs y length",
        ));
    }

    if n == 0 {
        return Err(RustyStatsError::EmptyInput("y is empty".to_string()));
    }

    if p == 0 {
        return Err(RustyStatsError::EmptyInput("X has no columns".to_string()));
    }

    let offset_vec = match offset {
        Some(o) => {
            if o.len() != n {
                return Err(RustyStatsError::dim_mismatch(
                    n,
                    o.len(),
                    "offset vs y length",
                ));
            }
            o.clone()
        }
        None => Array1::zeros(n),
    };

    let prior_weights_vec = match weights {
        Some(w) => {
            if w.len() != n {
                return Err(RustyStatsError::dim_mismatch(
                    n,
                    w.len(),
                    "weights vs y length",
                ));
            }
            if w.iter().any(|&x| x < 0.0) {
                return Err(RustyStatsError::InvalidValue(
                    "weights must be non-negative".to_string(),
                ));
            }
            w.clone()
        }
        None => Array1::ones(n),
    };

    Ok(ValidatedInputs {
        offset: offset_vec,
        prior_weights: prior_weights_vec,
    })
}

// =============================================================================
// Shared IRLS Weight Computation
// =============================================================================

/// Result of IRLS weight computation for a single iteration.
pub(crate) struct IRLSWeightResult {
    pub irls_weights: Array1<f64>,
    pub combined_weights: Array1<f64>,
    pub working_response: Array1<f64>,
}

/// Compute IRLS weights, combined weights, and working response in a single parallel pass.
///
/// Supports both Fisher information and true Hessian weighting for Gamma/Tweedie with log link.
pub(crate) fn compute_irls_weights(
    y: &Array1<f64>,
    mu: &Array1<f64>,
    eta: &Array1<f64>,
    offset: &Array1<f64>,
    prior_weights: &Array1<f64>,
    family: &dyn Family,
    link: &dyn Link,
    min_weight: f64,
) -> IRLSWeightResult {
    let n = y.len();
    let link_deriv = link.derivative(mu);

    let use_true_hessian = family.use_true_hessian_weights() && link.name() == "log";
    let hessian_weights = if use_true_hessian {
        Some(family.true_hessian_weights(mu, y))
    } else {
        None
    };
    let variance = if use_true_hessian {
        None
    } else {
        Some(family.variance(mu))
    };

    let results: Vec<(f64, f64, f64)> = (0..n)
        .into_par_iter()
        .map(|i| {
            let d = link_deriv[i];

            let iw = if let Some(ref hw) = hessian_weights {
                hw[i].max(min_weight).min(MAX_IRLS_WEIGHT)
            } else {
                let v = variance
                    .as_ref()
                    .expect("variance present in Fisher branch")[i];
                (1.0 / (v * d * d)).max(min_weight).min(MAX_IRLS_WEIGHT)
            };

            let cw = prior_weights[i] * iw;
            let wr = (eta[i] - offset[i]) + (y[i] - mu[i]) * d;

            (iw, cw, wr)
        })
        .collect();

    let mut irls_weights_vec = Vec::with_capacity(n);
    let mut combined_weights_vec = Vec::with_capacity(n);
    let mut working_response_vec = Vec::with_capacity(n);
    for (iw, cw, wr) in results {
        irls_weights_vec.push(iw);
        combined_weights_vec.push(cw);
        working_response_vec.push(wr);
    }

    IRLSWeightResult {
        irls_weights: Array1::from_vec(irls_weights_vec),
        combined_weights: Array1::from_vec(combined_weights_vec),
        working_response: Array1::from_vec(working_response_vec),
    }
}
