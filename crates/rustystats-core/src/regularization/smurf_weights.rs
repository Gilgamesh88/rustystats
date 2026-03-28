// =============================================================================
// SMuRF — Pesos adaptativos de penalización
// =============================================================================
//
// Porta desde R (.compute_pen_weights en Penalty_weights.R):
//   compute_penalty_weights()  — dispatcher principal
//   weights_equal()            — estrategia "eq"  (todos = 1)
//   weights_standardized()     — estrategia "stand" (Bondell & Reich, 2009)
//   weights_adaptive_glm()     — estrategia "glm"  (1/|GLM_coef|)
//   normalize_weights()        — normalización final compartida
//
// Fuera de scope (requiere GAM auxiliar): "gam", "gam.stand"
// =============================================================================

use ndarray::{Array1, ArrayView2};

use crate::error::{Result, RustyStatsError};
use crate::families::Family;
use crate::links::Link;
use crate::regularization::smurf_matrices::PenaltyMatrix;
use crate::regularization::smurf_types::{PenWeightsStrategy, SmurfTermSpec};
use crate::solvers::{fit_glm_unified, FitConfig, IRLSResult};

// =============================================================================
// Entry point principal
// =============================================================================

/// Calcula los pesos de penalización para todos los predictores.
///
/// Devuelve un Vec de Array1 — uno por predictor, con longitud igual
/// al número de filas de su matriz D (o n_params para Lasso/GroupLasso).
///
/// Los pesos se usan en el operador proximal:
///   slambda_i = lambda * step * pen_weight_i
pub fn compute_penalty_weights(
    strategy: &PenWeightsStrategy,
    terms: &[SmurfTermSpec],
    pen_matrices: &[Option<PenaltyMatrix>],
    x: ArrayView2<f64>,
    y: &Array1<f64>,
    weights: Option<&Array1<f64>>,
    offset: Option<&Array1<f64>>,
    family: &dyn Family,
    link: &dyn Link,
    lambda1: &mut Vec<Array1<f64>>,
    lambda2: &mut Vec<f64>,
    lambda1_orig: f64,
    lambda2_orig: f64,
) -> Result<Vec<Array1<f64>>> {
    let mut pen_weights = match strategy {
        PenWeightsStrategy::Eq => {
            weights_equal(terms, pen_matrices)
        }
        PenWeightsStrategy::Stand => {
            weights_standardized(terms, pen_matrices, x, y.len())
        }
        PenWeightsStrategy::Glm | PenWeightsStrategy::GlmStand => {
            let beta = fit_glm_for_weights(x, y, weights, offset, family, link)?;
            let raw = weights_adaptive_glm(terms, pen_matrices, &beta)?;
            if *strategy == PenWeightsStrategy::GlmStand {
                let stand = weights_standardized(terms, pen_matrices, x, y.len());
                // Combinar: glm * stand (igual que en R "glm.stand")
                raw.into_iter()
                    .zip(stand.into_iter())
                    .map(|(g, s)| g * s)
                    .collect()
            } else {
                raw
            }
        }
    };

    // Poner a cero los pesos de predictores sin penalización
    for (i, term) in terms.iter().enumerate() {
        if term.penalty_type.is_none() {
            pen_weights[i].fill(0.0);
        }
    }

    // Normalización final (equivale a Map("*", pen.weights, (n-1)/sum) en R)
    normalize_weights(
        &mut pen_weights,
        lambda1,
        lambda2,
        lambda1_orig,
        lambda2_orig,
    );

    Ok(pen_weights)
}

// =============================================================================
// Estrategia Eq — todos los pesos = 1
// =============================================================================

pub fn weights_equal(
    terms: &[SmurfTermSpec],
    pen_matrices: &[Option<PenaltyMatrix>],
) -> Vec<Array1<f64>> {
    terms
        .iter()
        .zip(pen_matrices.iter())
        .map(|(term, pen_mat)| {
            let n_rows = pen_mat
                .as_ref()
                .map(|d| d.nrows())
                .unwrap_or(term.n_params);
            Array1::ones(n_rows)
        })
        .collect()
}

// =============================================================================
// Estrategia Stand — Bondell & Reich (2009)
// =============================================================================
//
// Para cada fila k de D_j:
//   w_jk = sqrt( Σ_i |D_jk_i| * n_i / n ) * adj_factor
//
// donde n_i = número de observaciones en la categoría i,
// y adj_factor compensa por el tipo de penalización.
//
// Para Lasso/GroupLasso: w = 1 (no hay estandarización).
// Para FusedLasso: adj_factor = 1.
// Para GenFused/GraphGuided: adj_factor = n_params / n_rows(D).

pub fn weights_standardized(
    terms: &[SmurfTermSpec],
    pen_matrices: &[Option<PenaltyMatrix>],
    x: ArrayView2<f64>,
    n: usize,
) -> Vec<Array1<f64>> {
    terms
        .iter()
        .zip(pen_matrices.iter())
        .enumerate()
        .map(|(j, (term, pen_mat))| {
            use crate::regularization::smurf_types::SmurfPenaltyType;

            // Lasso y GroupLasso no necesitan estandarización
            match &term.penalty_type {
                SmurfPenaltyType::None
                | SmurfPenaltyType::Lasso
                | SmurfPenaltyType::GroupLasso { .. } => {
                    let n_rows = pen_mat
                        .as_ref()
                        .map(|d| d.nrows())
                        .unwrap_or(term.n_params);
                    return Array1::ones(n_rows);
                }
                _ => {}
            }

            let d = match pen_mat.as_ref() {
                Some(d) => d,
                None => {
                    return Array1::ones(term.n_params);
                }
            };

            let n_rows = d.nrows();
            let n_cols = d.ncols();

            // Calcular conteo de observaciones por nivel del predictor j
            // Las columnas de X que corresponden al predictor j
            let col_start: usize = terms[..j].iter().map(|t| t.n_params).sum();
            let col_end = col_start + term.n_params;

            let mut n_level = Array1::<f64>::zeros(n_cols);

            // Para predictores con ref_cat, el primer nivel es la referencia
            // (no aparece en X) — contamos las observaciones restantes
            if let Some(rc) = term.ref_cat {
                // Observaciones de referencia = n - suma del resto
                let mut sum_rest = 0.0f64;
                for col in col_start..col_end.min(x.ncols()) {
                    let col_sum: f64 = x.column(col).iter().sum();
                    sum_rest += col_sum;
                    let level_idx = col - col_start + if rc == 0 { 1 } else { 0 };
                    if level_idx < n_cols {
                        n_level[level_idx] = col_sum;
                    }
                }
                n_level[0] = (n as f64) - sum_rest;
            } else {
                for col in col_start..col_end.min(x.ncols()) {
                    let col_sum: f64 = x.column(col).iter().sum();
                    let level_idx = col - col_start;
                    if level_idx < n_cols {
                        n_level[level_idx] = col_sum;
                    }
                }
            }

            // Factor de ajuste por tipo de penalización
            let adj_factor = match &term.penalty_type {
                SmurfPenaltyType::FusedLasso => 1.0,
                _ => term.n_params as f64 / n_rows as f64,
            };

            // w_k = sqrt( Σ_i |D[k,i]| * n_i / n ) * adj_factor
            let mut weights = Array1::<f64>::zeros(n_rows);
            for row in 0..n_rows {
                let mut sum = 0.0f64;
                // Iterar no-ceros de la fila (recorriendo columnas de D)
                for col in 0..d.ncols() {
                    let rows = d.row_idx_of_col_raw(col);
                    let vals = d.val_of_col(col);
                    for (&r, &v) in rows.iter().zip(vals.iter()) {
                        if r == row {
                            sum += v.abs() * n_level[col];
                        }
                    }
                }
                weights[row] = (sum / n as f64).sqrt() * adj_factor;
                // Evitar pesos cero que causarían división por cero en adaptive
                if weights[row] < 1e-12 {
                    weights[row] = 1e-12;
                }
            }
            weights
        })
        .collect()
}

// =============================================================================
// Estrategia Glm — adaptive weights (1/|GLM_coef|)
// =============================================================================
//
// 1. Ajustar GLM sin penalización → beta_weights
// 2. Dividir beta_weights por predictor: beta_j = beta[offset_j..offset_j+n_params_j]
// 3. pen_weight_jk = 1 / |D_j * beta_j|_k   (por fila de D_j)
//
// Para GroupLasso: pen_weight = 1 / ||D_j * beta_j||_2

pub fn weights_adaptive_glm(
    terms: &[SmurfTermSpec],
    pen_matrices: &[Option<PenaltyMatrix>],
    beta: &Array1<f64>,
) -> Result<Vec<Array1<f64>>> {
    use crate::regularization::smurf_types::SmurfPenaltyType;

    // beta[0] es el intercepto — los términos empiezan en beta[1]
    let offsets: Vec<usize> = {
        let mut v = Vec::with_capacity(terms.len());
        let mut off = 1usize; // +1 por intercepto
        for term in terms {
            v.push(off);
            off += term.n_params;
        }
        v
    };

    terms
        .iter()
        .zip(pen_matrices.iter())
        .enumerate()
        .map(|(j, (term, pen_mat))| {
            let start = offsets[j];
            let end = (start + term.n_params).min(beta.len());
            let beta_j = beta.slice(ndarray::s![start..end]).to_owned();

            match &term.penalty_type {
                SmurfPenaltyType::None => {
                    let n_rows = pen_mat.as_ref().map(|d| d.nrows()).unwrap_or(term.n_params);
                    Ok(Array1::zeros(n_rows))
                }

                SmurfPenaltyType::Lasso => {
                    // w_i = 1 / |beta_j[i]|
                    let w: Array1<f64> = beta_j.mapv(|b| {
                        let ab = b.abs();
                        if ab < 1e-10 { 1e10 } else { 1.0 / ab }
                    });
                    Ok(w)
                }

                SmurfPenaltyType::GroupLasso { .. } => {
                    // w = 1 / ||beta_j||_2
                    let norm = beta_j.iter().map(|b| b * b).sum::<f64>().sqrt();
                    let w_val = if norm < 1e-10 { 1e10 } else { 1.0 / norm };
                    let n_rows = pen_mat.as_ref().map(|d| d.nrows()).unwrap_or(term.n_params);
                    Ok(Array1::from_elem(n_rows, w_val))
                }

                // Para Fused y variantes: w_k = 1 / |D_j * beta_j|_k
                _ => {
                    let d = pen_mat.as_ref().ok_or_else(|| {
                        RustyStatsError::MissingParameter(format!(
                            "Término '{}' necesita pen_matrix para pesos adaptativos",
                            term.name
                        ))
                    })?;

                    let n_rows = d.nrows();
                    let mut d_beta = Array1::<f64>::zeros(n_rows);

                    // D_j * beta_j (mat-vec sparse)
                    for col in 0..d.ncols().min(beta_j.len()) {
                        let rows = d.row_idx_of_col_raw(col);
                        let vals = d.val_of_col(col);
                        for (&row, &dval) in rows.iter().zip(vals.iter()) {
                            d_beta[row] += dval * beta_j[col];
                        }
                    }

                    // w_k = 1 / |D_j * beta_j|_k
                    let w: Array1<f64> = d_beta.mapv(|v| {
                        let av = v.abs();
                        if av < 1e-10 { 1e10 } else { 1.0 / av }
                    });
                    Ok(w)
                }
            }
        })
        .collect()
}

// =============================================================================
// Normalización final
// =============================================================================
//
// Equivale a Map("*", pen.weights, (length(tmp)-1)/sum(tmp)) en R.
// Normaliza para que la suma de todos los pesos sea n_weights - 1.
// También escala lambda1 y lambda2 con el mismo factor.

pub fn normalize_weights(
    pen_weights: &mut Vec<Array1<f64>>,
    lambda1: &mut Vec<Array1<f64>>,
    lambda2: &mut Vec<f64>,
    lambda1_orig: f64,
    lambda2_orig: f64,
) {
    // Suma total de todos los pesos
    let total: f64 = pen_weights.iter().map(|w| w.sum()).sum();

    // Suma de lambda1 pesos (solo para predictores con Fused/Graph)
    let lambda1_sum: f64 = if lambda1_orig > 0.0 {
        lambda1.iter().map(|l| l.sum()).sum::<f64>() / lambda1_orig
    } else {
        0.0
    };

    // Suma de lambda2 pesos
    let lambda2_sum: f64 = if lambda2_orig > 0.0 {
        lambda2.iter().sum::<f64>() / lambda2_orig
    } else {
        0.0
    };

    let n_weights = pen_weights.iter().map(|w| w.len()).sum::<usize>() as f64
        + if lambda1_orig > 0.0 { lambda1_sum } else { 0.0 }
        + if lambda2_orig > 0.0 { lambda2_sum } else { 0.0 };

    let sum_all = total
        + if lambda1_orig > 0.0 { lambda1.iter().map(|l| l.sum()).sum::<f64>() } else { 0.0 }
        + if lambda2_orig > 0.0 { lambda2.iter().sum::<f64>() } else { 0.0 };

    if sum_all < 1e-12 {
        return; // Todos los pesos son cero — nada que normalizar
    }

    let factor = (n_weights - 1.0) / sum_all;

    for w in pen_weights.iter_mut() {
        w.mapv_inplace(|v| v * factor);
    }
    for l1 in lambda1.iter_mut() {
        l1.mapv_inplace(|v| v * factor);
    }
    for l2 in lambda2.iter_mut() {
        *l2 *= factor;
    }
}

// =============================================================================
// Helper: ajustar GLM sin penalización para pesos adaptativos
// =============================================================================

fn fit_glm_for_weights(
    x: ArrayView2<f64>,
    y: &Array1<f64>,
    weights: Option<&Array1<f64>>,
    offset: Option<&Array1<f64>>,
    family: &dyn Family,
    link: &dyn Link,
) -> Result<Array1<f64>> {
    let config = FitConfig::default();

    let result: IRLSResult = fit_glm_unified(
        y,
        x,
        family,
        link,
        &config,
        offset,
        weights,
        None,
    )?;

    Ok(result.coefficients)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::regularization::smurf_matrices::build_fused_lasso_matrix;
    use crate::regularization::smurf_types::{SmurfPenaltyType, SmurfTermSpec};
    use ndarray::array;

    fn make_lasso_term() -> SmurfTermSpec {
        SmurfTermSpec::lasso("x", 3)
    }

    fn make_fused_term() -> SmurfTermSpec {
        SmurfTermSpec::fused_lasso("age", 4, Some(0))
    }

    #[test]
    fn test_weights_equal_lasso() {
        let term = make_lasso_term();
        let d = build_fused_lasso_matrix(3, None).unwrap();
        let weights = weights_equal(&[term], &[Some(d)]);
        // Lasso: n_rows de D = 3-1 = 2... pero usamos n_params=3 para identidad
        // En equal, usamos d.nrows()
        assert!(weights[0].iter().all(|&w| (w - 1.0).abs() < 1e-10));
    }

    #[test]
    fn test_weights_equal_all_ones() {
        let terms = vec![
            SmurfTermSpec::lasso("a", 2),
            SmurfTermSpec::fused_lasso("b", 3, None),
        ];
        let d_a = build_fused_lasso_matrix(2, None).unwrap(); // 1×2
        let d_b = build_fused_lasso_matrix(3, None).unwrap(); // 2×3
        let weights = weights_equal(&terms, &[Some(d_a), Some(d_b)]);
        assert!(weights[0].iter().all(|&w| (w - 1.0).abs() < 1e-10));
        assert!(weights[1].iter().all(|&w| (w - 1.0).abs() < 1e-10));
    }

    #[test]
    fn test_normalize_weights_sum() {
        let mut pen_weights = vec![
            Array1::from_vec(vec![2.0, 4.0]),
            Array1::from_vec(vec![1.0, 3.0]),
        ];
        let mut lambda1 = vec![Array1::zeros(2), Array1::zeros(2)];
        let mut lambda2 = vec![0.0, 0.0];

        normalize_weights(&mut pen_weights, &mut lambda1, &mut lambda2, 0.0, 0.0);

        // Suma total de pesos debe ser n_weights - 1 = 4 - 1 = 3
        let total: f64 = pen_weights.iter().map(|w| w.sum()).sum();
        assert!((total - 3.0).abs() < 1e-8, "suma={}", total);
    }

    #[test]
    fn test_normalize_weights_proportions_preserved() {
        let mut pen_weights = vec![
            Array1::from_vec(vec![1.0, 1.0]),
            Array1::from_vec(vec![1.0, 1.0]),
        ];
        let mut lambda1 = vec![Array1::zeros(2), Array1::zeros(2)];
        let mut lambda2 = vec![0.0, 0.0];

        normalize_weights(&mut pen_weights, &mut lambda1, &mut lambda2, 0.0, 0.0);

        // Todos iguales → todos deben ser 3/4 = 0.75
        for w in &pen_weights {
            for &v in w.iter() {
                assert!((v - 0.75).abs() < 1e-8, "w={}", v);
            }
        }
    }

    #[test]
    fn test_weights_adaptive_glm_lasso() {
        // beta = [intercept=0, b1=2.0, b2=0.5, b3=4.0]
        let beta = array![0.0, 2.0, 0.5, 4.0];
        let term = SmurfTermSpec::lasso("x", 3);
        let d = build_fused_lasso_matrix(3, None).unwrap();
        let weights = weights_adaptive_glm(&[term], &[Some(d)], &beta).unwrap();

        // w[0] = 1/|2.0| = 0.5
        // w[1] = 1/|0.5| = 2.0
        // w[2] = 1/|4.0| = 0.25
        assert!((weights[0][0] - 0.5).abs() < 1e-8, "w0={}", weights[0][0]);
        assert!((weights[0][1] - 2.0).abs() < 1e-8, "w1={}", weights[0][1]);
        assert!((weights[0][2] - 0.25).abs() < 1e-8, "w2={}", weights[0][2]);
    }

    #[test]
    fn test_weights_adaptive_glm_zero_coef() {
        // Coeficiente cero → peso muy grande (capped a 1e10)
        let beta = array![0.0, 0.0, 1.0, 2.0];
        let term = SmurfTermSpec::lasso("x", 3);
        let d = build_fused_lasso_matrix(3, None).unwrap();
        let weights = weights_adaptive_glm(&[term], &[Some(d)], &beta).unwrap();
        assert!(weights[0][0] > 1e9, "peso de coef cero debe ser grande");
    }

    #[test]
    fn test_weights_adaptive_glm_fused() {
        // Para FusedLasso, pesos = 1/|D*beta|
        // D (2×3): fila0 = [-1, 1, 0], fila1 = [0, -1, 1]
        // beta_j = [b1=1.0, b2=3.0, b3=3.0]
        // D*beta = [3-1, 3-3] = [2.0, 0.0]
        // w = [1/2, 1e10]
        let beta = array![0.0, 1.0, 3.0, 3.0]; // intercept + 3 coefs
        let term = SmurfTermSpec::fused_lasso("age", 3, None);
        let d = build_fused_lasso_matrix(3, None).unwrap();
        let weights = weights_adaptive_glm(&[term], &[Some(d)], &beta).unwrap();
        assert!((weights[0][0] - 0.5).abs() < 1e-8, "w0={}", weights[0][0]);
        assert!(weights[0][1] > 1e9, "D*beta[1]=0 → peso grande");
    }

    #[test]
    fn test_compute_penalty_weights_eq_strategy() {
        use crate::families::GaussianFamily;
        use crate::links::IdentityLink;

        let terms = vec![SmurfTermSpec::lasso("x", 2)];
        let d = build_fused_lasso_matrix(2, None).unwrap(); // 1×2
        let pen_matrices = vec![Some(d)];
        let mut lambda1 = vec![Array1::zeros(1)];
        let mut lambda2 = vec![0.0];

        let x = array![[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]];
        let y = array![1.0, 2.0, 3.0];
        let family = GaussianFamily;
        let link = IdentityLink;

        let weights = compute_penalty_weights(
            &PenWeightsStrategy::Eq,
            &terms,
            &pen_matrices,
            x.view(),
            &y,
            None,
            None,
            &family,
            &link,
            &mut lambda1,
            &mut lambda2,
            0.0,
            0.0,
        ).unwrap();

        assert_eq!(weights.len(), 1);
        // Después de normalización, los pesos deben ser positivos
        assert!(weights[0].iter().all(|&w| w >= 0.0));
    }
}
