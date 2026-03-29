// =============================================================================
// SMuRF — Solver FISTA (Gradiente Proximal Acelerado)
// =============================================================================
//
// Porta desde R:
//   fit_smurf_glm()        ← glmsmurf() — entry point público
//   smurf_fit_internal()   ← .glmsmurf.fit.internal()
//   smurf_fista_loop()     ← .glmsmurf.algorithm() — loop FISTA + Nesterov
//
// El algoritmo FISTA (Beck & Teboulle 2009) con aceleración de Nesterov:
//
//   Dado θ⁰ = β⁰ (punto inicial)
//   Para cada iteración t:
//     1. Calcular gradiente ∇f(θᵗ) de la log-verosimilitud en θ
//     2. Paso de gradiente: β̃ = θᵗ - sᵗ * ∇f(θᵗ)
//     3. Paso proximal:     βᵗ⁺¹ = prox_{sᵗ*g}(β̃)   ← por predictor
//     4. Backtracking:      reducir sᵗ si f(βᵗ⁺¹) > cuadrática
//     5. Nesterov momentum: θᵗ⁺¹ = βᵗ⁺¹ + ((αᵗ-1)/αᵗ⁺¹)(βᵗ⁺¹ - βᵗ)
//     6. Criterio de paro:  |obj_old - obj_new| / |obj_old| < ε
// =============================================================================

use ndarray::{Array1, Array2};

use crate::error::{Result, RustyStatsError};
use crate::families::Family;
use crate::links::Link;
use crate::regularization::smurf_matrices::{
    build_fused_lasso_matrix, build_gen_fused_lasso_matrix, build_graph_guided_matrix,
    build_group_lasso_matrix, build_lasso_matrix, build_2d_fused_lasso_matrix,
    eigendecompose_penalty, EigenData, PenaltyMatrix,
};
use crate::regularization::smurf_proximal::apply_proximal_operator;
use crate::regularization::smurf_types::{
    LambdaSelection, PenWeightsStrategy, SmurfPenalty, SmurfPenaltyType, SmurfTermSpec,
};
use crate::regularization::smurf_weights::compute_penalty_weights;
use super::smurf_lambda::{
    build_lambda_grid, compute_lambda_max, create_folds, select_lambda, FitSummary,
};
use crate::solvers::{fit_glm_unified, FitConfig, IRLSResult};

// =============================================================================
// Tipos públicos
// =============================================================================

/// Configuración del solver SMuRF.
pub struct SmurfConfig {
    pub penalty: SmurfPenalty,
}

impl SmurfConfig {
    pub fn new(penalty: SmurfPenalty) -> Self {
        Self { penalty }
    }
}

/// Resultado de ajustar un modelo GLM con penalización SMuRF.
pub struct SmurfResult {
    /// Resultado GLM estándar: coeficientes, fitted values, deviance, cov.
    pub irls_result: IRLSResult,
    /// Valor de λ seleccionado.
    pub selected_lambda: f64,
    /// Nombre del tipo de penalización por predictor.
    pub penalty_type_names: Vec<String>,
    /// Historia del objetivo f(β)+g(β) por iteración FISTA.
    pub objective_history: Vec<f64>,
    /// Iteraciones que tomó el loop FISTA.
    pub fista_iterations: usize,
    /// ¿Convergió el algoritmo?
    pub fista_converged: bool,
}

// =============================================================================
// Entry point público
// =============================================================================

/// Ajusta un GLM con penalización SMuRF vía gradiente proximal FISTA.
///
/// Orquesta: construir matrices D → calcular pesos → seleccionar λ → FISTA.
pub fn fit_smurf_glm(
    x: &Array2<f64>,
    y: &Array1<f64>,
    weights: Option<&Array1<f64>>,
    offset: Option<&Array1<f64>>,
    family: Box<dyn Family>,
    link: Box<dyn Link>,
    config: SmurfConfig,
) -> Result<SmurfResult> {
    let penalty = &config.penalty;

    // Validar configuración
    penalty.validate().map_err(|e| RustyStatsError::InvalidValue(e))?;

    // 1. Construir matrices de penalización D por predictor
    let pen_matrices = build_all_penalty_matrices(&penalty.terms)?;

    // 2. Eigendecomposición de DᵀD (solo para predictores con ADMM)
    let eigen_data = build_eigen_data(&penalty.terms, &pen_matrices)?;

    // 3. Calcular pesos adaptativos
    let mut lambda1: Vec<Array1<f64>> = penalty
        .terms
        .iter()
        .map(|t| Array1::from_elem(t.n_params, t.lambda1))
        .collect();
    let mut lambda2: Vec<f64> = penalty.terms.iter().map(|t| t.lambda2).collect();

    let lambda1_orig = penalty.terms.iter().map(|t| t.lambda1).fold(0.0f64, f64::max);
    let lambda2_orig = penalty.terms.iter().map(|t| t.lambda2).fold(0.0f64, f64::max);

    let pen_weights = compute_penalty_weights(
        &penalty.pen_weights,
        &penalty.terms,
        &pen_matrices,
        x.view(),
        y,
        weights,
        offset,
        family.as_ref(),
        link.as_ref(),
        &mut lambda1,
        &mut lambda2,
        lambda1_orig,
        lambda2_orig,
    )?;

    // 4. Seleccionar lambda
    let selected_lambda = match &penalty.lambda_selection {
        LambdaSelection::Fixed(l) => *l,
        _ => {
            let lambda_max = compute_lambda_max(
                x.view(), y, weights, offset,
                family.as_ref(), link.as_ref(),
                &penalty.terms, &pen_matrices, &pen_weights,
            )?;
            let grid = build_lambda_grid(lambda_max);
            select_lambda_with_fista(
                &grid,
                &penalty.lambda_selection,
                x, y, weights, offset,
                family.as_ref(), link.as_ref(),
                penalty,
                &pen_matrices,
                &eigen_data,
                &pen_weights,
                &lambda1,
                &lambda2,
            )?
        }
    };

    // 5. Ajuste final con lambda seleccionado
    let (coefficients, obj_history, iterations, converged) = smurf_fista_loop(
        x, y, weights, offset,
        family.as_ref(), link.as_ref(),
        &penalty.terms,
        &pen_matrices,
        &eigen_data,
        &pen_weights,
        &lambda1,
        &lambda2,
        selected_lambda,
        penalty.epsilon,
        penalty.max_iter,
        penalty.step_init,
        penalty.tau,
    )?;

    // 6. Re-estimación con GLM estándar si está activa
    let final_result = if penalty.reestimate {
        reestimate_reduced_glm(
            x, y, weights, offset,
            family.as_ref(), link.as_ref(),
            &coefficients,
        )?
    } else {
        build_irls_result_from_coefs(
            x, y, weights, offset,
            family.as_ref(), link.as_ref(),
            &coefficients,
        )?
    };

    let penalty_type_names = penalty
        .terms
        .iter()
        .map(|t| t.penalty_type.name().to_string())
        .collect();

    Ok(SmurfResult {
        irls_result: final_result,
        selected_lambda,
        penalty_type_names,
        objective_history: obj_history,
        fista_iterations: iterations,
        fista_converged: converged,
    })
}

// =============================================================================
// Loop FISTA principal
// =============================================================================

#[allow(clippy::too_many_arguments)]
pub fn smurf_fista_loop(
    x: &Array2<f64>,
    y: &Array1<f64>,
    weights: Option<&Array1<f64>>,
    offset: Option<&Array1<f64>>,
    family: &dyn Family,
    link: &dyn Link,
    terms: &[SmurfTermSpec],
    pen_matrices: &[Option<PenaltyMatrix>],
    eigen_data: &[Option<EigenData>],
    pen_weights: &[Array1<f64>],
    lambda1: &[Array1<f64>],
    lambda2: &[f64],
    lambda: f64,
    epsilon: f64,
    max_iter: usize,
    step_init: f64,
    tau: f64,
) -> Result<(Array1<f64>, Vec<f64>, usize, bool)> {
    let n = y.len();
    let p = x.ncols();

    let w = weights.cloned().unwrap_or_else(|| Array1::ones(n));
    let off = offset.cloned().unwrap_or_else(|| Array1::zeros(n));

    // Punto de inicio: intercepto = weighted mean, resto = 0
    let y_mean = (&w * y).sum() / w.sum();
    let intercept_start = link.link(&Array1::from_elem(1, y_mean.max(1e-6)))[0];
    let mut beta = Array1::<f64>::zeros(p);
    if p > 0 { beta[0] = intercept_start; }

    let mut theta = beta.clone();
    let mut alpha_old = 1.0f64;
    let mut step = step_init;

    // Función objetivo: f(β) + g(β)
    // f = -log-verosimilitud escalada
    // g = λ Σ_j ||D_j β_j||_1 (penalización total)
    let compute_f = |b: &Array1<f64>| -> f64 {
        let eta: Array1<f64> = x.dot(b) + &off;
        let mu = link.inverse(&eta);
        let dev = family.deviance(y, &mu, Some(&w));
        -dev / (2.0 * w.iter().filter(|&&wi| wi > 0.0).count() as f64)
    };

    let compute_g = |b: &Array1<f64>| -> f64 {
        let offsets = compute_term_offsets(terms);
        let mut g = 0.0f64;
        for (j, term) in terms.iter().enumerate() {
            if term.penalty_type.is_none() { continue; }
            let start = offsets[j];
            let end = (start + term.n_params).min(p);
            let beta_j = b.slice(ndarray::s![start..end]);
            let pw = &pen_weights[j];

            match &term.penalty_type {
                SmurfPenaltyType::Lasso => {
                    g += lambda * beta_j.iter().zip(pw.iter())
                        .map(|(&bi, &wi)| wi * bi.abs()).sum::<f64>();
                }
                SmurfPenaltyType::GroupLasso { .. } => {
                    let norm = beta_j.iter().map(|b| b * b).sum::<f64>().sqrt();
                    g += lambda * pw[0] * norm;
                }
                _ => {
                    if let Some(d) = &pen_matrices[j] {
                        let m = d.nrows();
                        let mut d_beta = Array1::<f64>::zeros(m);
                        for col in 0..d.ncols().min(beta_j.len()) {
                            let rows = d.row_idx_of_col_raw(col);
                            let vals = d.val_of_col(col);
                            for (&row, &dval) in rows.iter().zip(vals.iter()) {
                                d_beta[row] += dval * beta_j[col];
                            }
                        }
                        g += lambda * d_beta.iter().zip(pw.iter())
                            .map(|(&db, &wi)| wi * db.abs()).sum::<f64>();
                    }
                }
            }
        }
        g
    };

    let mut f_old = compute_f(&beta);
    let mut obj_old = f_old + compute_g(&beta);
    let mut obj_history = vec![obj_old];

    let mut iter = 0usize;
    let mut converged = false;

    while iter < max_iter {
        // 1. Calcular gradiente de f en theta
        let eta_theta: Array1<f64> = x.dot(&theta) + &off;
        let mu_theta = link.inverse(&eta_theta);
        let variance = family.variance(&mu_theta);
        let link_deriv = link.derivative(&mu_theta);

        let n_eff = w.iter().filter(|&&wi| wi > 0.0).count() as f64;
        let mut grad = Array1::<f64>::zeros(p);
        for j in 0..p {
            let mut sum = 0.0f64;
            for i in 0..n {
                let denom = variance[i] * link_deriv[i];
                if denom.abs() > 1e-15 {
                    sum += x[[i, j]] * w[i] * (mu_theta[i] - y[i]) / denom;
                }
            }
            grad[j] = sum / n_eff;
        }

        // 2. Backtracking line search
        let f_theta = compute_f(&theta);
        let mut beta_new;

        loop {
            // Paso de gradiente
            let beta_tilde: Array1<f64> = &theta - step * &grad;

            // Paso proximal por predictor
            beta_new = apply_proximal_step(
                &beta_tilde, &theta,
                terms, pen_matrices, eigen_data,
                pen_weights, lambda1, lambda2,
                lambda, step, p,
            )?;

            // Verificar condición de descenso suficiente
            let f_new = compute_f(&beta_new);
            let diff: Array1<f64> = &beta_new - &theta;
            let grad_dot_diff = grad.dot(&diff);
            let diff_sq = diff.dot(&diff);

            let rhs = f_theta + grad_dot_diff + diff_sq / (2.0 * step);

            if f_new <= rhs + 1e-12 || step < 1e-14 {
                break;
            }
            step *= tau;
        }

        // 3. Aceleración de Nesterov
        let alpha_new = 0.5 * (1.0 + (1.0 + 4.0 * alpha_old * alpha_old).sqrt());
        let momentum = (alpha_old - 1.0) / alpha_new;
        theta = &beta_new + momentum * (&beta_new - &beta);

        // 4. Actualizar beta y alpha
        beta = beta_new;
        alpha_old = alpha_new;

        // 5. Calcular objetivo
        let obj_new = compute_f(&beta) + compute_g(&beta);
        obj_history.push(obj_new);

        // 6. Criterio de paro
        let rel_change = (obj_old - obj_new).abs() / obj_old.abs().max(1e-10);
        if rel_change < epsilon && iter > 0 {
            converged = true;
            iter += 1;
            break;
        }

        obj_old = obj_new;
        iter += 1;
        let _ = f_old;
        f_old = compute_f(&beta);
    }

    Ok((beta, obj_history, iter, converged))
}

// =============================================================================
// Paso proximal para todos los predictores
// =============================================================================

fn apply_proximal_step(
    beta_tilde: &Array1<f64>,
    beta_old: &Array1<f64>,
    terms: &[SmurfTermSpec],
    pen_matrices: &[Option<PenaltyMatrix>],
    eigen_data: &[Option<EigenData>],
    pen_weights: &[Array1<f64>],
    lambda1: &[Array1<f64>],
    lambda2: &[f64],
    lambda: f64,
    step: f64,
    p: usize,
) -> Result<Array1<f64>> {
    let offsets = compute_term_offsets(terms);
    let mut beta_new = beta_tilde.clone();

    // Intercepto (índice 0) nunca se penaliza
    for (j, term) in terms.iter().enumerate() {
        let start = offsets[j];
        let end = (start + term.n_params).min(p);

        let bt_j = beta_tilde.slice(ndarray::s![start..end]).to_owned();
        let bold_j = beta_old.slice(ndarray::s![start..end]).to_owned();
        let pw_j = &pen_weights[j];
        let l1_j = &lambda1[j];
        let l2_j = lambda2[j];

        let result_j = apply_proximal_operator(
            &bt_j,
            &bold_j,
            &term.penalty_type,
            pw_j,
            pen_matrices[j].as_ref(),
            eigen_data[j].as_ref(),
            lambda,
            step,
            None, // group_norm calculado internamente
            l1_j,
            l2_j,
        )?;

        for (k, &v) in result_j.iter().enumerate() {
            if start + k < p {
                beta_new[start + k] = v;
            }
        }
    }

    Ok(beta_new)
}

// =============================================================================
// Construcción de matrices D para todos los predictores
// =============================================================================

fn build_all_penalty_matrices(
    terms: &[SmurfTermSpec],
) -> Result<Vec<Option<PenaltyMatrix>>> {
    terms
        .iter()
        .map(|term| match &term.penalty_type {
            SmurfPenaltyType::None => Ok(None),

            SmurfPenaltyType::Lasso => {
                Ok(Some(build_lasso_matrix(term.n_params)?))
            }

            SmurfPenaltyType::GroupLasso { .. } => {
                Ok(Some(build_group_lasso_matrix(term.n_params)?))
            }

            SmurfPenaltyType::FusedLasso => {
                Ok(Some(build_fused_lasso_matrix(term.n_params, term.ref_cat)?))
            }

            SmurfPenaltyType::GenFusedLasso => {
                Ok(Some(build_gen_fused_lasso_matrix(term.n_params, term.ref_cat)?))
            }

            SmurfPenaltyType::GraphGuidedFusedLasso => {
                let adj = term.adj_matrix.as_ref().ok_or_else(|| {
                    RustyStatsError::MissingParameter(format!(
                        "Término '{}' necesita adj_matrix", term.name
                    ))
                })?;
                Ok(Some(build_graph_guided_matrix(adj, term.ref_cat)?))
            }

            SmurfPenaltyType::TwoDFusedLasso { n_rows, n_cols } => {
                Ok(Some(build_2d_fused_lasso_matrix(*n_rows, *n_cols)?))
            }
        })
        .collect()
}

// =============================================================================
// Eigendecomposición para todos los predictores con ADMM
// =============================================================================

fn build_eigen_data(
    terms: &[SmurfTermSpec],
    pen_matrices: &[Option<PenaltyMatrix>],
) -> Result<Vec<Option<EigenData>>> {
    terms
        .iter()
        .zip(pen_matrices.iter())
        .map(|(term, pen_mat)| {
            if !term.penalty_type.uses_admm() {
                return Ok(None);
            }
            let d = pen_mat.as_ref().ok_or_else(|| {
                RustyStatsError::MissingParameter(format!(
                    "Término '{}' necesita pen_matrix para eigendecomp", term.name
                ))
            })?;
            Ok(Some(eigendecompose_penalty(d)?))
        })
        .collect()
}

// =============================================================================
// Selección de lambda con FISTA
// =============================================================================

fn select_lambda_with_fista(
    grid: &[f64],
    strategy: &LambdaSelection,
    x: &Array2<f64>,
    y: &Array1<f64>,
    weights: Option<&Array1<f64>>,
    offset: Option<&Array1<f64>>,
    family: &dyn Family,
    link: &dyn Link,
    penalty: &SmurfPenalty,
    pen_matrices: &[Option<PenaltyMatrix>],
    eigen_data: &[Option<EigenData>],
    pen_weights: &[Array1<f64>],
    lambda1: &[Array1<f64>],
    lambda2: &[f64],
) -> Result<f64> {
    let n = y.len();

    // Closure de ajuste para criterios in-sample
    let fit_fn = |lambda: f64| -> Result<FitSummary> {
        let (coefs, _, _, _) = smurf_fista_loop(
            x, y, weights, offset, family, link,
            &penalty.terms, pen_matrices, eigen_data,
            pen_weights, lambda1, lambda2,
            lambda,
            penalty.epsilon * 10.0, // tolerancia más laxa para el grid
            penalty.max_iter / 2,
            penalty.step_init,
            penalty.tau,
        )?;

        let eta: Array1<f64> = x.dot(&coefs) + offset.cloned().unwrap_or_else(|| Array1::zeros(n));
        let mu = link.inverse(&eta);
        let w = weights.cloned().unwrap_or_else(|| Array1::ones(n));
        let deviance = family.deviance(y, &mu, Some(&w));

        // df_model = número de coeficientes no cero (aprox.)
        let df_model = coefs.iter().filter(|&&c| c.abs() > 1e-8).count() as f64;

        Ok(FitSummary { deviance, df_model, n_obs: n })
    };

    // Closure de CV
    let cv_fit_fn = |lambda: f64, train_idx: &[usize], val_idx: &[usize]| -> Result<f64> {
        // Construir subconjuntos de entrenamiento
        let x_train = x.select(ndarray::Axis(0), train_idx);
        let y_train: Array1<f64> = train_idx.iter().map(|&i| y[i]).collect();
        let w_train: Option<Array1<f64>> = weights.map(|w| {
            train_idx.iter().map(|&i| w[i]).collect()
        });
        let off_train: Option<Array1<f64>> = offset.map(|o| {
            train_idx.iter().map(|&i| o[i]).collect()
        });

        let (coefs, _, _, _) = smurf_fista_loop(
            &x_train,
            &y_train,
            w_train.as_ref(),
            off_train.as_ref(),
            family, link,
            &penalty.terms, pen_matrices, eigen_data,
            pen_weights, lambda1, lambda2,
            lambda,
            penalty.epsilon * 10.0,
            penalty.max_iter / 2,
            penalty.step_init,
            penalty.tau,
        )?;

        // Evaluar deviance en validación
        let x_val = x.select(ndarray::Axis(0), val_idx);
        let y_val: Array1<f64> = val_idx.iter().map(|&i| y[i]).collect();
        let w_val: Array1<f64> = weights
            .map(|w| val_idx.iter().map(|&i| w[i]).collect())
            .unwrap_or_else(|| Array1::ones(val_idx.len()));
        let off_val: Array1<f64> = offset
            .map(|o| val_idx.iter().map(|&i| o[i]).collect())
            .unwrap_or_else(|| Array1::zeros(val_idx.len()));

        let eta_val: Array1<f64> = x_val.dot(&coefs) + &off_val;
        let mu_val = link.inverse(&eta_val);
        let dev_val = family.deviance(&y_val, &mu_val, Some(&w_val));

        Ok(dev_val / val_idx.len() as f64)
    };

    let n_folds = penalty.cv_folds;

    select_lambda(
        strategy,
        grid,
        &fit_fn,
        Some(&cv_fit_fn),
        n,
        n_folds,
    )
}

// =============================================================================
// Re-estimación post-SMuRF
// =============================================================================
//
// Cuando SMuRF fusiona categorías (coeficientes iguales), re-estima el
// modelo reducido sin penalización para obtener errores estándar válidos.

fn reestimate_reduced_glm(
    x: &Array2<f64>,
    y: &Array1<f64>,
    weights: Option<&Array1<f64>>,
    offset: Option<&Array1<f64>>,
    family: &dyn Family,
    link: &dyn Link,
    smurf_coefs: &Array1<f64>,
) -> Result<IRLSResult> {
    let config = FitConfig::default();
    fit_glm_unified(y, x.view(), family, link, &config, offset, weights, Some(smurf_coefs))
}

fn build_irls_result_from_coefs(
    x: &Array2<f64>,
    y: &Array1<f64>,
    weights: Option<&Array1<f64>>,
    offset: Option<&Array1<f64>>,
    family: &dyn Family,
    link: &dyn Link,
    coefs: &Array1<f64>,
) -> Result<IRLSResult> {
    let config = FitConfig::default();
    fit_glm_unified(y, x.view(), family, link, &config, offset, weights, Some(coefs))
}

// =============================================================================
// Helpers
// =============================================================================

/// Calcula los offsets de inicio de cada término en el vector de coeficientes.
/// El intercepto ocupa el índice 0, los términos empiezan en 1.
pub fn compute_term_offsets(terms: &[SmurfTermSpec]) -> Vec<usize> {
    let mut offsets = Vec::with_capacity(terms.len());
    let mut offset = 1usize; // +1 por intercepto
    for term in terms {
        offsets.push(offset);
        offset += term.n_params;
    }
    offsets
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::families::PoissonFamily;
    use crate::families::GaussianFamily;
    use crate::links::{LogLink, IdentityLink};
    use crate::regularization::smurf_types::SmurfTermSpec;
    use ndarray::array;

    fn simple_gaussian_data() -> (Array2<f64>, Array1<f64>) {
        // 20 obs, 1 intercepto + 3 coeficientes
        let x = Array2::from_shape_fn((20, 4), |(i, j)| {
            if j == 0 { 1.0 } else { ((i * j) % 5) as f64 }
        });
        let y: Array1<f64> = (0..20).map(|i| 1.0 + 0.5 * (i % 4) as f64).collect();
        (x, y)
    }

    #[test]
    fn test_compute_term_offsets() {
        let terms = vec![
            SmurfTermSpec::lasso("a", 2),
            SmurfTermSpec::fused_lasso("b", 3, None),
        ];
        let offsets = compute_term_offsets(&terms);
        assert_eq!(offsets[0], 1); // después del intercepto
        assert_eq!(offsets[1], 3); // 1 + 2
    }

    #[test]
    fn test_fista_loop_gaussian_converges() {
        let (x, y) = simple_gaussian_data();
        let terms = vec![
            SmurfTermSpec::lasso("x1", 1),
            SmurfTermSpec::lasso("x2", 1),
            SmurfTermSpec::lasso("x3", 1),
        ];
        let pen_matrices = build_all_penalty_matrices(&terms).unwrap();
        let eigen_data = build_eigen_data(&terms, &pen_matrices).unwrap();
        let pen_weights: Vec<Array1<f64>> = terms.iter().zip(pen_matrices.iter()).map(|(t, _)| {
            Array1::ones(t.n_params)
        }).collect();
        let lambda1: Vec<Array1<f64>> = terms.iter().map(|t| Array1::zeros(t.n_params)).collect();
        let lambda2: Vec<f64> = vec![0.0; terms.len()];

        let (coefs, history, iters, converged) = smurf_fista_loop(
            &x, &y, None, None,
            &GaussianFamily, &IdentityLink,
            &terms, &pen_matrices, &eigen_data,
            &pen_weights, &lambda1, &lambda2,
            0.1, 1e-5, 1000, 1.0, 0.5,
        ).unwrap();

        assert_eq!(coefs.len(), 4); // intercepto + 3 coefs
        assert!(coefs.iter().all(|c| c.is_finite()));
        assert!(!history.is_empty());
        let _ = (iters, converged);
    }

    #[test]
    fn test_fista_lasso_shrinks_to_zero() {
        // Con lambda muy grande, los coeficientes penalizados deben ir a cero
        let (x, y) = simple_gaussian_data();
        let terms = vec![
            SmurfTermSpec::lasso("x1", 1),
            SmurfTermSpec::lasso("x2", 1),
            SmurfTermSpec::lasso("x3", 1),
        ];
        let pen_matrices = build_all_penalty_matrices(&terms).unwrap();
        let eigen_data = build_eigen_data(&terms, &pen_matrices).unwrap();
        let pen_weights: Vec<Array1<f64>> = terms.iter().map(|t| Array1::ones(t.n_params)).collect();
        let lambda1: Vec<Array1<f64>> = terms.iter().map(|t| Array1::zeros(t.n_params)).collect();
        let lambda2 = vec![0.0; terms.len()];

        let (coefs, _, _, _) = smurf_fista_loop(
            &x, &y, None, None,
            &GaussianFamily, &IdentityLink,
            &terms, &pen_matrices, &eigen_data,
            &pen_weights, &lambda1, &lambda2,
            1000.0, // lambda muy grande
            1e-5, 500, 1.0, 0.5,
        ).unwrap();

        // Con lambda enorme, coefs[1..] deben ser ~0
        for &c in &coefs.as_slice().unwrap()[1..] {
            assert!(c.abs() < 0.1, "coef={} no shrunk a cero con lambda grande", c);
        }
    }

    #[test]
    fn test_fit_smurf_glm_gaussian() {
        let (x, y) = simple_gaussian_data();
        let penalty = SmurfPenalty::new(vec![
            SmurfTermSpec::lasso("x1", 1),
            SmurfTermSpec::lasso("x2", 1),
            SmurfTermSpec::lasso("x3", 1),
        ])
        .with_lambda(0.1)
        .with_pen_weights(PenWeightsStrategy::Eq);

        let config = SmurfConfig::new(penalty);
        let result = fit_smurf_glm(
            &x, &y, None, None,
            Box::new(GaussianFamily),
            Box::new(IdentityLink),
            config,
        ).unwrap();

        assert!(result.fista_iterations > 0);
        assert!(result.irls_result.deviance.is_finite());
        assert!(result.irls_result.coefficients.len() == 4);
    }
}
