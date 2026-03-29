// =============================================================================
// SMuRF — Selección del parámetro de penalización λ
// =============================================================================
//
// Porta desde R (.select.lambda, .max.lambda en Lambda_select.R y Lambda_max.R):
//   compute_lambda_max()   ← .max.lambda()
//   build_lambda_grid()    ← grid logarítmico sobre [lambda_max*1e-4, lambda_max]
//   select_lambda()        ← .select.lambda() — CV e in-sample
//
// La selección de λ es necesaria porque SMuRF requiere λ > 0 pero el valor
// óptimo depende de los datos. Lambda_max es el λ por encima del cual todos
// los coeficientes penalizados se vuelven cero.
// =============================================================================

use ndarray::{Array1, ArrayView2};

use crate::error::{Result, RustyStatsError};
use crate::families::Family;
use crate::links::Link;
use crate::regularization::smurf_matrices::PenaltyMatrix;
use crate::regularization::smurf_types::{LambdaSelection, SmurfPenaltyType, SmurfTermSpec};

// =============================================================================
// Constantes
// =============================================================================

/// Número de lambdas en el grid de búsqueda.
const N_LAMBDAS: usize = 30;

/// Ratio lambda_min / lambda_max (en escala log).
const LAMBDA_RATIO: f64 = 1e-4;

// =============================================================================
// Lambda máximo
// =============================================================================
//
// El lambda máximo es el valor por encima del cual todos los coeficientes
// penalizados son exactamente cero. Se usa como límite superior del grid.
//
// Para Lasso:
//   λ_max_j = max_i( |∇f(0)_i| / w_i )
//   donde ∇f(0) = gradiente de la log-verosimilitud en β=0
//
// Para GroupLasso:
//   λ_max_j = ||∇f(0)_j / w_j||_2
//
// Para FusedLasso / variantes:
//   Aproximación: max fila de |D_j * ∇f(0)_j| / w_k
//   (La solución exacta requiere el algoritmo PPF de Earle et al. 2017,
//    que se puede añadir en v2 si se necesita precisión mayor.)

pub fn compute_lambda_max(
    x: ArrayView2<f64>,
    y: &Array1<f64>,
    weights: Option<&Array1<f64>>,
    offset: Option<&Array1<f64>>,
    family: &dyn Family,
    link: &dyn Link,
    terms: &[SmurfTermSpec],
    pen_matrices: &[Option<PenaltyMatrix>],
    pen_weights: &[Array1<f64>],
) -> Result<f64> {
    let n = y.len();
    let p = x.ncols();

    // Pesos prior
    let w: Array1<f64> = weights.cloned().unwrap_or_else(|| Array1::ones(n));

    // Offset
    let off: Array1<f64> = offset.cloned().unwrap_or_else(|| Array1::zeros(n));

    // Punto de inicio: beta = 0 → eta = offset, mu = linkinv(offset)
    let eta: Array1<f64> = off.clone();
    let mu = link.inverse(&eta);

    // Gradiente de -log-verosimilitud en beta=0:
    // ∇f(β)_j = Σ_i x_ij * w_i * (μ_i - y_i) / (V(μ_i) * g'(μ_i)) / n_eff
    let n_eff = w.iter().filter(|&&wi| wi > 0.0).count() as f64;
    let variance = family.variance(&mu);
    let link_deriv = link.derivative(&mu);

    let mut grad = Array1::<f64>::zeros(p);
    for j in 0..p {
        let mut sum = 0.0f64;
        for i in 0..n {
            let denom = variance[i] * link_deriv[i];
            if denom.abs() > 1e-15 {
                sum += x[[i, j]] * w[i] * (mu[i] - y[i]) / denom;
            }
        }
        grad[j] = sum / n_eff;
    }

    // Calcular lambda_max por predictor
    let offsets: Vec<usize> = {
        let mut v = Vec::with_capacity(terms.len());
        let mut off_idx = 1usize; // +1 por intercepto
        for term in terms {
            v.push(off_idx);
            off_idx += term.n_params;
        }
        v
    };

    let mut lambda_max_vals: Vec<f64> = Vec::new();

    for (j, term) in terms.iter().enumerate() {
        if term.penalty_type.is_none() {
            continue;
        }

        let start = offsets[j];
        let end = (start + term.n_params).min(p);
        let grad_j: Array1<f64> = grad.slice(ndarray::s![start..end]).to_owned();
        let pw = &pen_weights[j];

        let lmax_j = match &term.penalty_type {
            SmurfPenaltyType::None => continue,

            SmurfPenaltyType::Lasso => {
                // max_i( |∇f_i| / w_i )
                grad_j
                    .iter()
                    .zip(pw.iter())
                    .map(|(&g, &w)| {
                        if w > 1e-15 { g.abs() / w } else { 0.0 }
                    })
                    .fold(0.0f64, f64::max)
            }

            SmurfPenaltyType::GroupLasso { .. } => {
                // ||∇f_j / w||_2  (w es escalar para GroupLasso)
                let w_scalar = pw[0].max(1e-15);
                grad_j
                    .iter()
                    .map(|&g| (g / w_scalar).powi(2))
                    .sum::<f64>()
                    .sqrt()
            }

            // Para Fused y variantes: aproximación via |grad|
            SmurfPenaltyType::FusedLasso
            | SmurfPenaltyType::GenFusedLasso
            | SmurfPenaltyType::TwoDFusedLasso { .. } => {
                if let Some(d) = pen_matrices[j].as_ref() {
                    // Aproximación: max fila de |D * grad_j| / w_k
                    let m = d.nrows();
                    let mut d_grad = Array1::<f64>::zeros(m);
                    for col in 0..d.ncols().min(grad_j.len()) {
                        let rows = d.row_idx_of_col_raw(col);
                        let vals = d.val_of_col(col);
                        for (&row, &dval) in rows.iter().zip(vals.iter()) {
                            d_grad[row] += dval * grad_j[col];
                        }
                    }
                    d_grad
                        .iter()
                        .zip(pw.iter())
                        .map(|(&dg, &w)| {
                            if w > 1e-15 { dg.abs() / w } else { 0.0 }
                        })
                        .fold(0.0f64, f64::max)
                } else {
                    grad_j.iter().map(|g| g.abs()).fold(0.0f64, f64::max)
                }
            }

            SmurfPenaltyType::GraphGuidedFusedLasso => {
                if let Some(d) = pen_matrices[j].as_ref() {
                    let m = d.nrows();
                    let mut d_grad = Array1::<f64>::zeros(m);
                    for col in 0..d.ncols().min(grad_j.len()) {
                        let rows = d.row_idx_of_col_raw(col);
                        let vals = d.val_of_col(col);
                        for (&row, &dval) in rows.iter().zip(vals.iter()) {
                            d_grad[row] += dval * grad_j[col];
                        }
                    }
                    d_grad
                        .iter()
                        .zip(pw.iter())
                        .map(|(&dg, &w)| {
                            if w > 1e-15 { dg.abs() / w } else { dg.abs() }
                        })
                        .fold(0.0f64, f64::max)
                } else {
                    0.0
                }
            }
        };

        if lmax_j > 0.0 {
            lambda_max_vals.push(lmax_j);
        }
    }

    if lambda_max_vals.is_empty() {
        return Err(RustyStatsError::InvalidValue(
            "No se pudo calcular lambda_max — ningún predictor tiene penalización activa".to_string()
        ));
    }

    // Lambda global = máximo sobre todos los predictores
    Ok(lambda_max_vals.iter().cloned().fold(0.0f64, f64::max))
}

// =============================================================================
// Grid logarítmico de lambdas
// =============================================================================
//
// Devuelve N_LAMBDAS valores en escala log entre lambda_max y lambda_max*RATIO.
// Orden: de mayor a menor (como en glmnet y glmsmurf en R).

pub fn build_lambda_grid(lambda_max: f64) -> Vec<f64> {
    let log_max = lambda_max.ln();
    let log_min = (lambda_max * LAMBDA_RATIO).ln();
    let step = (log_max - log_min) / (N_LAMBDAS - 1) as f64;

    (0..N_LAMBDAS)
        .map(|i| (log_max - i as f64 * step).exp())
        .collect()
}

// =============================================================================
// Selección de lambda
// =============================================================================
//
// Estrategias in-sample: evalúan el criterio (AIC/BIC/GCV) con todos los datos.
// Estrategias CV: k-fold cross-validación, minimizar deviance media.
//
// Nota: para poder llamar al solver FISTA necesitamos una función de ajuste
// que se pasa como closure — evita dependencia circular con solvers/smurf.rs.


/// Resumen mínimo de un ajuste SMuRF para selección de lambda.
pub struct FitSummary {
    /// Deviance del modelo ajustado
    pub deviance: f64,
    /// Número efectivo de parámetros (df del modelo)
    pub df_model: f64,
    /// Número de observaciones
    pub n_obs: usize,
}

/// Selecciona lambda usando la estrategia especificada.
///
/// `fit_fn`: closure que ajusta el modelo para un lambda dado y devuelve FitSummary.
/// `lambda_grid`: grid de lambdas a evaluar (de mayor a menor).
/// `y_full`: vector respuesta completo (para CV).
/// `n_folds`: número de folds para CV.
pub fn select_lambda(
    strategy: &LambdaSelection,
    lambda_grid: &[f64],
    fit_fn: &dyn Fn(f64) -> Result<FitSummary>,
    cv_fit_fn: Option<&dyn Fn(f64, &[usize], &[usize]) -> Result<f64>>,
    n_obs: usize,
    n_folds: usize,
) -> Result<f64> {
    match strategy {
        LambdaSelection::Fixed(lambda) => Ok(*lambda),

        LambdaSelection::IsAic => {
            select_insample(lambda_grid, fit_fn, |s| {
                s.deviance + 2.0 * s.df_model
            })
        }

        LambdaSelection::IsBic => {
            select_insample(lambda_grid, fit_fn, |s| {
                s.deviance + (n_obs as f64).ln() * s.df_model
            })
        }

        LambdaSelection::IsGcv => {
            select_insample(lambda_grid, fit_fn, |s| {
                let n = s.n_obs as f64;
                let denom = (1.0 - s.df_model / n).powi(2);
                if denom < 1e-10 { f64::INFINITY } else { s.deviance / n / denom }
            })
        }

        LambdaSelection::CvDev | LambdaSelection::CvMse => {
            let cv_fn = cv_fit_fn.ok_or_else(|| {
                RustyStatsError::MissingParameter(
                    "CV requiere cv_fit_fn".to_string()
                )
            })?;
            select_cv(lambda_grid, cv_fn, n_obs, n_folds, false)
        }

        LambdaSelection::CvOneSeDev => {
            let cv_fn = cv_fit_fn.ok_or_else(|| {
                RustyStatsError::MissingParameter(
                    "CV 1SE requiere cv_fit_fn".to_string()
                )
            })?;
            select_cv(lambda_grid, cv_fn, n_obs, n_folds, true)
        }
    }
}

// =============================================================================
// Helpers internos
// =============================================================================

/// Selección in-sample: evalúa criterion(fit) en cada lambda, devuelve el mínimo.
fn select_insample(
    lambda_grid: &[f64],
    fit_fn: &dyn Fn(f64) -> Result<FitSummary>,
    criterion: impl Fn(&FitSummary) -> f64,
) -> Result<f64> {
    let mut best_lambda = lambda_grid[0];
    let mut best_crit = f64::INFINITY;

    for &lambda in lambda_grid {
        match fit_fn(lambda) {
            Ok(summary) => {
                let crit = criterion(&summary);
                if crit < best_crit {
                    best_crit = crit;
                    best_lambda = lambda;
                }
            }
            Err(_) => continue, // Ignorar lambdas que no convergen
        }
    }

    Ok(best_lambda)
}

/// Selección por CV: k-fold, minimiza deviance media.
/// Si `one_se_rule = true`, selecciona el lambda más grande dentro de
/// 1 error estándar del mínimo (modelo más parsimonioso).
fn select_cv(
    lambda_grid: &[f64],
    cv_fit_fn: &dyn Fn(f64, &[usize], &[usize]) -> Result<f64>,
    n_obs: usize,
    n_folds: usize,
    one_se_rule: bool,
) -> Result<f64> {
    // Crear folds: índices para cada fold
    let folds = create_folds(n_obs, n_folds);

    let mut mean_deviances = Vec::with_capacity(lambda_grid.len());
    let mut se_deviances = Vec::with_capacity(lambda_grid.len());

    for &lambda in lambda_grid {
        let mut fold_deviances = Vec::with_capacity(n_folds);

        for fold_idx in 0..n_folds {
            // Índices de validación = fold actual
            let val_idx: Vec<usize> = folds[fold_idx].clone();
            // Índices de entrenamiento = todos los demás folds
            let train_idx: Vec<usize> = (0..n_folds)
                .filter(|&f| f != fold_idx)
                .flat_map(|f| folds[f].iter().copied())
                .collect();

            match cv_fit_fn(lambda, &train_idx, &val_idx) {
                Ok(dev) => fold_deviances.push(dev),
                Err(_) => {
                    // Si falla un fold, marcar este lambda como malo
                    fold_deviances.push(f64::INFINITY);
                }
            }
        }

        let mean = fold_deviances.iter().sum::<f64>() / n_folds as f64;
        let se = {
            let var = fold_deviances
                .iter()
                .map(|&d| (d - mean).powi(2))
                .sum::<f64>()
                / (n_folds as f64 - 1.0);
            (var / n_folds as f64).sqrt()
        };

        mean_deviances.push(mean);
        se_deviances.push(se);
    }

    // Encontrar el lambda con deviance mínima
    let min_idx = mean_deviances
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);

    if !one_se_rule {
        return Ok(lambda_grid[min_idx]);
    }

    // Regla 1-SE: buscar el lambda más grande (menos regularización)
    // cuya deviance está dentro de mean_min + se_min
    let threshold = mean_deviances[min_idx] + se_deviances[min_idx];
    let best_idx = mean_deviances
        .iter()
        .enumerate()
        .take(min_idx + 1) // Solo lambdas más grandes (índices menores)
        .filter(|(_, &d)| d <= threshold)
        .map(|(i, _)| i)
        .next()
        .unwrap_or(min_idx);

    Ok(lambda_grid[best_idx])
}

/// Crea k folds de índices aproximadamente iguales.
pub fn create_folds(n: usize, k: usize) -> Vec<Vec<usize>> {
    let mut folds = vec![Vec::new(); k];
    for i in 0..n {
        folds[i % k].push(i);
    }
    folds
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::regularization::smurf_matrices::{build_fused_lasso_matrix, build_lasso_matrix};
    use crate::regularization::smurf_types::SmurfTermSpec;
    use ndarray::array;

    #[test]
    fn test_build_lambda_grid_length() {
        let grid = build_lambda_grid(1.0);
        assert_eq!(grid.len(), N_LAMBDAS);
    }

    #[test]
    fn test_build_lambda_grid_decreasing() {
        let grid = build_lambda_grid(1.0);
        for i in 1..grid.len() {
            assert!(grid[i] < grid[i-1], "grid debe ser decreciente");
        }
    }

    #[test]
    fn test_build_lambda_grid_bounds() {
        let lambda_max = 2.0;
        let grid = build_lambda_grid(lambda_max);
        assert!((grid[0] - lambda_max).abs() < 1e-10, "primer valor = lambda_max");
        assert!(grid.last().unwrap() >= &(lambda_max * LAMBDA_RATIO * 0.99));
    }

    #[test]
    fn test_create_folds_coverage() {
        let folds = create_folds(10, 3);
        let mut all_idx: Vec<usize> = folds.iter().flatten().copied().collect();
        all_idx.sort();
        assert_eq!(all_idx, (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn test_create_folds_count() {
        let folds = create_folds(10, 5);
        assert_eq!(folds.len(), 5);
        let total: usize = folds.iter().map(|f| f.len()).sum();
        assert_eq!(total, 10);
    }

    #[test]
    fn test_select_lambda_fixed() {
        let result = select_lambda(
            &LambdaSelection::Fixed(0.42),
            &[1.0, 0.5, 0.1],
            &|_| Ok(FitSummary { deviance: 1.0, df_model: 2.0, n_obs: 100 }),
            None,
            100,
            5,
        ).unwrap();
        assert!((result - 0.42).abs() < 1e-10);
    }

    #[test]
    fn test_select_lambda_aic_picks_minimum() {
        // AIC = deviance + 2*df
        // lambda=1.0 → dev=10, df=5 → AIC=20
        // lambda=0.5 → dev=8,  df=4 → AIC=16  ← mínimo
        // lambda=0.1 → dev=9,  df=3 → AIC=15  ← mínimo real
        let grid = vec![1.0, 0.5, 0.1];
        let deviances = vec![10.0, 8.0, 9.0];
        let dfs = vec![5.0, 4.0, 3.0];

        let result = select_lambda(
            &LambdaSelection::IsAic,
            &grid,
            &|lambda| {
                let i = grid.iter().position(|&l| (l - lambda).abs() < 1e-10).unwrap();
                Ok(FitSummary { deviance: deviances[i], df_model: dfs[i], n_obs: 100 })
            },
            None,
            100,
            5,
        ).unwrap();

        // AIC: 20, 16, 15 → mínimo en lambda=0.1
        assert!((result - 0.1).abs() < 1e-10, "lambda={}", result);
    }

    #[test]
    fn test_compute_lambda_max_positive() {
        use crate::families::PoissonFamily;
        use crate::links::LogLink;

        let x = array![
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        ];
        let y = array![2.0, 1.0, 3.0, 4.0];
        let terms = vec![
            SmurfTermSpec::lasso("a", 1),
            SmurfTermSpec::lasso("b", 1),
        ];
        let d1 = build_lasso_matrix(1).unwrap();
        let d2 = build_lasso_matrix(1).unwrap();
        let pen_matrices = vec![Some(d1), Some(d2)];
        let pen_weights = vec![
            Array1::ones(1),
            Array1::ones(1),
        ];

        let family = PoissonFamily;
        let link = LogLink;

        let lmax = compute_lambda_max(
            x.view(), &y, None, None,
            &family, &link,
            &terms, &pen_matrices, &pen_weights,
        ).unwrap();

        assert!(lmax > 0.0, "lambda_max debe ser positivo: {}", lmax);
    }
}
