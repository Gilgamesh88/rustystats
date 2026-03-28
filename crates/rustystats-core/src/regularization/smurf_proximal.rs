// =============================================================================
// SMuRF — Operadores proximales + ADMM (faer sparse)
// =============================================================================
//
// Porta desde R / C++:
//   apply_proximal_operator()  ← .PO() dispatcher
//   proximal_lasso()           ← .PO.Lasso()
//   proximal_group_lasso()     ← .PO.GroupLasso()
//   proximal_admm_fused()      ← admm_po_cpp() en Penalty_PO.cpp
// =============================================================================

use ndarray::Array1;

use crate::error::{Result, RustyStatsError};
use crate::regularization::smurf_matrices::{EigenData, PenaltyMatrix};
use crate::regularization::smurf_types::SmurfPenaltyType;
use crate::regularization::soft_threshold;

// =============================================================================
// Dispatcher principal
// =============================================================================

/// Aplica el operador proximal correcto según el tipo de penalización.
///
/// Llamado para cada predictor en cada iteración del loop FISTA.
/// `beta_tilde` = coeficientes después del paso de gradiente (θ - s∇f).
/// Devuelve β actualizado para el predictor j.
pub fn apply_proximal_operator(
    beta_tilde: &Array1<f64>,
    beta_old: &Array1<f64>,
    penalty_type: &SmurfPenaltyType,
    pen_weights: &Array1<f64>, // pesos por fila de D (o por coef para Lasso)
    pen_matrix: Option<&PenaltyMatrix>,
    eigen: Option<&EigenData>,
    lambda: f64,
    step: f64,
    // Para GroupLasso: norma del bloque completo del grupo
    group_norm: Option<f64>,
    // Para FusedLasso sparse: lambda1 y lambda2 adicionales
    lambda1: &Array1<f64>,
    lambda2: f64,
) -> Result<Array1<f64>> {
    match penalty_type {
        SmurfPenaltyType::None => Ok(beta_tilde.clone()),

        SmurfPenaltyType::Lasso => {
            // slambda por coeficiente = lambda * step * pen_weight[i]
            let slambda: Array1<f64> = pen_weights.mapv(|w| lambda * step * w);
            Ok(proximal_lasso(beta_tilde, &slambda))
        }

        SmurfPenaltyType::GroupLasso { .. } => {
            let norm = group_norm.unwrap_or_else(|| {
                beta_tilde.iter().map(|x| x * x).sum::<f64>().sqrt()
            });
            // slambda = lambda * step * pen_weight (primer elemento del bloque)
            let slambda = lambda * step * pen_weights[0];
            Ok(proximal_group_lasso(beta_tilde, slambda, norm))
        }

        SmurfPenaltyType::FusedLasso
        | SmurfPenaltyType::GenFusedLasso
        | SmurfPenaltyType::GraphGuidedFusedLasso
        | SmurfPenaltyType::TwoDFusedLasso { .. } => {
            let d = pen_matrix.ok_or_else(|| {
                RustyStatsError::MissingParameter(
                    "ADMM requiere pen_matrix para Fused/GraphGuided".to_string(),
                )
            })?;
            let eigen_data = eigen.ok_or_else(|| {
                RustyStatsError::MissingParameter(
                    "ADMM requiere EigenData precalculado".to_string(),
                )
            })?;
            let slambda = lambda * step;
            proximal_admm_fused(
                beta_tilde,
                beta_old,
                d,
                eigen_data,
                pen_weights,
                slambda,
                lambda1,
                lambda2,
                10_000,
                1.0,
            )
        }
    }
}

// =============================================================================
// Operador proximal — Lasso
// =============================================================================
//
// Soft-thresholding elemento a elemento con umbral variable por coeficiente.
// Reutiliza soft_threshold() de regularization/mod.rs.
//
// prox_{s*λ*||·||₁}(z)_i = sign(z_i) * max(|z_i| - slambda_i, 0)

pub fn proximal_lasso(beta_tilde: &Array1<f64>, slambda: &Array1<f64>) -> Array1<f64> {
    beta_tilde
        .iter()
        .zip(slambda.iter())
        .map(|(&b, &sl)| soft_threshold(b, sl))
        .collect()
}

// =============================================================================
// Operador proximal — Group Lasso
// =============================================================================
//
// Shrinkage de bloque: escala todo el vector por max(1 - slambda/||β||, 0).
// La norma se calcula sobre el bloque completo del grupo (puede incluir
// múltiples predictores si group_id > 0).
//
// prox_{s*λ*||·||₂}(z) = z * max(1 - slambda/||z||, 0)

pub fn proximal_group_lasso(
    beta_tilde: &Array1<f64>,
    slambda: f64,
    norm: f64,
) -> Array1<f64> {
    if norm <= 0.0 {
        return Array1::zeros(beta_tilde.len());
    }
    let scale = (1.0 - slambda / norm).max(0.0);
    beta_tilde.mapv(|b| b * scale)
}

// =============================================================================
// Operador proximal — ADMM para Fused Lasso y variantes
// =============================================================================
//
// Porta directamente admm_po_cpp() de Penalty_PO.cpp en smurf-master.
//
// Resuelve: prox_{s*λ*||D·||₁}(β̃)
// mediante el algoritmo ADMM con relajación (Boyd et al., 2011).
//
// Variables duales:
//   x ∈ ℝⁿ  — estimado primal (coeficientes)
//   z ∈ ℝᵐ  — variable auxiliar (= Dx en el óptimo)
//   u ∈ ℝᵐ  — multiplicador de Lagrange escalado
//
// Actualización de x (paso rápido con eigendecomp):
//   x = (I + ρ DᵀD)⁻¹ (β̃ + ρ Dᵀ(z - u))
//     = β̃ + ρ Dᵀ(z-u) - ρ Q diag(1/(eigvals+ρ)) Qᵀ [β̃ + ρ Dᵀ(z-u)]
//
// Paso lento (cuando hay eigvals ≈ 0):
//   x = inv(I + ρ DᵀD) * (β̃ + ρ Dᵀ(z - u))
//
// Actualización de z (soft-threshold):
//   z = S_{slambda/ρ}(D*x_relaxed + u)
//
// Post-procesado (Sparse Fused Lasso, si lambda1 > 0 o lambda2 > 0):
//   x = group_soft_thresh(x, slambda*lambda2)   si lambda2 > 0
//   x = soft_thresh(x, slambda*lambda1)          si lambda1 > 0

pub fn proximal_admm_fused(
    beta_tilde: &Array1<f64>,
    beta_old: &Array1<f64>,
    d: &PenaltyMatrix,
    eigen: &EigenData,
    pen_weights: &Array1<f64>, // pesos por fila de D
    slambda: f64,
    lambda1: &Array1<f64>,
    lambda2: f64,
    max_iter: usize,
    rho_init: f64,
) -> Result<Array1<f64>> {
    let m = d.nrows(); // número de constraints (filas de D)
    let n = d.ncols(); // número de coeficientes

    // Tolerancias (igual que en C++)
    let eps_rel = 1e-10_f64;
    let eps_abs = 1e-12_f64;

    // Parámetros de actualización adaptativa de ρ (Zhu 2017, JCGS)
    let mu_rho = 10.0_f64;
    let eta = 2.0_f64;
    let xi = 1.5_f64; // relajación

    // Inicialización
    let mut x = Array1::<f64>::zeros(n);
    let mut z_old = Array1::<f64>::zeros(m);
    let mut z_new = mat_vec_mul(d, beta_old); // D * beta_old como punto de inicio
    let mut u = Array1::<f64>::zeros(m);
    let mut rho = rho_init;

    // Matriz auxiliar precalculada (se actualiza si ρ cambia)
    // fast: ADMM_aux * v = v - ρ Q diag(1/(eigvals+ρ)) Qᵀ v
    // slow: ADMM_aux = inv(I + ρ DᵀD)
    let mut admm_aux = build_admm_aux(d, eigen, rho, n)?;

    let mut iter = 0usize;

    loop {
        z_old.clone_from(&z_new);

        // 1. Actualizar x
        //    rhs = β̃ + ρ Dᵀ(z_old - u)
        let z_minus_u: Array1<f64> = &z_old - &u;
        let dt_zm_u = mat_t_vec_mul(d, &z_minus_u); // Dᵀ (z - u)
        let rhs: Array1<f64> = beta_tilde + rho * &dt_zm_u;
        x = apply_admm_aux(&admm_aux, &rhs, d, eigen, rho, n)?;

        // 2. Relajación: x_hat = ξ D x + (1-ξ) z_old
        let dx = mat_vec_mul(d, &x);
        let x_hat: Array1<f64> = dx.mapv(|v| xi * v) + z_old.mapv(|v| (1.0 - xi) * v);

        // 3. Actualizar z: soft-threshold con pesos
        //    z_new = S_{slambda*w/ρ}(x_hat + u)
        let x_hat_plus_u: Array1<f64> = &x_hat + &u;
        z_new = soft_threshold_weighted(&x_hat_plus_u, pen_weights, slambda / rho);

        // 4. Actualizar u
        u = &u + &x_hat - &z_new;

        // 5. Residuales y tolerancias
        let primal_res = mat_vec_mul(d, &x);
        let r_norm = norm2(&(&primal_res - &z_new));
        let s_norm = rho * norm2(&mat_t_vec_mul(d, &(&z_new - &z_old)));

        let eps_pri = (m as f64).sqrt() * eps_abs
            + eps_rel * norm2(&primal_res).max(norm2(&z_new));
        let eps_dual = (n as f64).sqrt() * eps_abs
            + eps_rel * rho * norm2(&mat_t_vec_mul(d, &u));

        iter += 1;

        // 6. Verificar convergencia
        if (r_norm <= eps_pri && s_norm <= eps_dual) || iter >= max_iter {
            break;
        }

        // 7. Actualización adaptativa de ρ
        if r_norm / eps_pri >= mu_rho * s_norm / eps_dual {
            let rho_old = rho;
            rho *= eta;
            u.mapv_inplace(|v| v * rho_old / rho);
            admm_aux = build_admm_aux(d, eigen, rho, n)?;
        } else if s_norm / eps_dual >= mu_rho * r_norm / eps_pri {
            let rho_old = rho;
            rho /= eta;
            u.mapv_inplace(|v| v * rho_old / rho);
            admm_aux = build_admm_aux(d, eigen, rho, n)?;
        }
    }

    // Post-procesado: Sparse Fused Lasso (Group component)
    if lambda2 > 0.0 {
        let norm_x = norm2(&x);
        x = proximal_group_lasso(&x, slambda * lambda2, norm_x);
    }

    // Post-procesado: Sparse Fused Lasso (L1 component)
    if lambda1.iter().any(|&v| v > 0.0) {
        x = x
            .iter()
            .zip(lambda1.iter())
            .map(|(&xi, &l1)| soft_threshold(xi, slambda * l1))
            .collect();
    }

    Ok(x)
}

// =============================================================================
// Helpers internos
// =============================================================================

/// Enum para la matriz auxiliar del ADMM.
/// Fast: almacena Q y eigvals para el producto eficiente.
/// Slow: almacena la inversa densa.
enum AdmmAux {
    Fast {
        q: faer::Mat<f64>,
        scaled_eigvals: Vec<f64>, // 1/(eigvals + ρ)
        rho: f64,
    },
    Slow {
        inv_matrix: ndarray::Array2<f64>, // (I + ρ DᵀD)⁻¹
    },
}

/// Construye la matriz auxiliar del ADMM según el modo (fast/slow).
fn build_admm_aux(
    d: &PenaltyMatrix,
    eigen: &EigenData,
    rho: f64,
    n: usize,
) -> Result<AdmmAux> {
    if eigen.fast_mode {
        let scaled_eigvals: Vec<f64> = eigen
            .eigvals
            .iter()
            .map(|&ev| 1.0 / (ev + rho))
            .collect();
        Ok(AdmmAux::Fast {
            q: eigen.q.clone(),
            scaled_eigvals,
            rho,
        })
    } else {
        // Construir I + ρ DᵀD como ndarray y luego invertir con nalgebra
        let dtd = compute_dtd_dense(d, n);
        let inv = invert_i_plus_rho_dtd(&dtd, rho, n)?;
        Ok(AdmmAux::Slow { inv_matrix: inv })
    }
}

/// Aplica la matriz auxiliar del ADMM a un vector rhs.
/// Fast: v - ρ Q diag(1/(eigvals+ρ)) Qᵀ v
/// Slow: inv_matrix * v
fn apply_admm_aux(
    aux: &AdmmAux,
    rhs: &Array1<f64>,
    _d: &PenaltyMatrix,
    _eigen: &EigenData,
    _rho: f64,
    n: usize,
) -> Result<Array1<f64>> {
    match aux {
        AdmmAux::Fast { q, scaled_eigvals, rho } => {
            // qt_rhs = Qᵀ * rhs
            let qt_rhs = mat_t_vec_faer(q, rhs, n);
            // scaled = diag(1/(eigvals+ρ)) * qt_rhs
            let scaled: Vec<f64> = qt_rhs
                .iter()
                .zip(scaled_eigvals.iter())
                .map(|(&v, &s)| v * s)
                .collect();
            // q_scaled = Q * scaled
            let q_scaled = mat_vec_faer(q, &scaled, n);
            // result = rhs - ρ * Q * scaled
            let result: Array1<f64> = rhs
                .iter()
                .zip(q_scaled.iter())
                .map(|(&r, &qs)| r - rho * qs)
                .collect();
            Ok(result)
        }
        AdmmAux::Slow { inv_matrix } => {
            // result = inv_matrix * rhs
            let result = inv_matrix.dot(rhs);
            Ok(result)
        }
    }
}

/// Soft-threshold con pesos por elemento (para pen_weights en ADMM).
fn soft_threshold_weighted(
    v: &Array1<f64>,
    weights: &Array1<f64>,
    base_threshold: f64,
) -> Array1<f64> {
    v.iter()
        .zip(weights.iter())
        .map(|(&vi, &wi)| soft_threshold(vi, base_threshold * wi))
        .collect()
}

/// Norma L2 de un vector ndarray.
fn norm2(v: &Array1<f64>) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// D * v (sparse mat-vec, D es SparseColMat CSC).
fn mat_vec_mul(d: &PenaltyMatrix, v: &Array1<f64>) -> Array1<f64> {
    let m = d.nrows();
    let n = d.ncols();
    let mut result = Array1::<f64>::zeros(m);

    for col in 0..n {
        let val_v = v[col];
        if val_v == 0.0 { continue; }
        let rows = d.row_idx_of_col_raw(col);
        let vals = d.val_of_col(col);
        for (&row, &dval) in rows.iter().zip(vals.iter()) {
            result[row] += dval * val_v;
        }
    }
    result
}

/// Dᵀ * v (sparse transpose mat-vec).
fn mat_t_vec_mul(d: &PenaltyMatrix, v: &Array1<f64>) -> Array1<f64> {
    let n = d.ncols();
    let mut result = Array1::<f64>::zeros(n);

    for col in 0..n {
        let rows = d.row_idx_of_col_raw(col);
        let vals = d.val_of_col(col);
        let mut dot = 0.0f64;
        for (&row, &dval) in rows.iter().zip(vals.iter()) {
            dot += dval * v[row];
        }
        result[col] = dot;
    }
    result
}

/// Q * v donde Q es faer::Mat<f64>, v es Vec<f64>, resultado en Array1.
fn mat_vec_faer(q: &faer::Mat<f64>, v: &[f64], n: usize) -> Array1<f64> {
    let mut result = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut sum = 0.0f64;
        for j in 0..n {
            sum += q[(i, j)] * v[j];
        }
        result[i] = sum;
    }
    result
}

/// Qᵀ * v donde Q es faer::Mat<f64>, v es Array1<f64>, resultado en Vec<f64>.
fn mat_t_vec_faer(q: &faer::Mat<f64>, v: &Array1<f64>, n: usize) -> Vec<f64> {
    let mut result = vec![0.0f64; n];
    for j in 0..n {
        let mut sum = 0.0f64;
        for i in 0..n {
            sum += q[(i, j)] * v[i];
        }
        result[j] = sum;
    }
    result
}

/// Calcula DᵀD como ndarray::Array2<f64> (para el path lento).
fn compute_dtd_dense(d: &PenaltyMatrix, n: usize) -> ndarray::Array2<f64> {
    let mut dtd = ndarray::Array2::<f64>::zeros((n, n));
    for col_i in 0..n {
        for col_j in col_i..n {
            let rows_i = d.row_idx_of_col_raw(col_i);
            let vals_i = d.val_of_col(col_i);
            let rows_j = d.row_idx_of_col_raw(col_j);
            let vals_j = d.val_of_col(col_j);
            let mut dot = 0.0f64;
            let mut pi = 0;
            let mut pj = 0;
            while pi < rows_i.len() && pj < rows_j.len() {
                match rows_i[pi].cmp(&rows_j[pj]) {
                    std::cmp::Ordering::Equal => {
                        dot += vals_i[pi] * vals_j[pj];
                        pi += 1; pj += 1;
                    }
                    std::cmp::Ordering::Less    => { pi += 1; }
                    std::cmp::Ordering::Greater => { pj += 1; }
                }
            }
            dtd[[col_i, col_j]] = dot;
            dtd[[col_j, col_i]] = dot;
        }
    }
    dtd
}

/// Invierte (I + ρ DᵀD) usando nalgebra (ya en el proyecto).
fn invert_i_plus_rho_dtd(
    dtd: &ndarray::Array2<f64>,
    rho: f64,
    n: usize,
) -> Result<ndarray::Array2<f64>> {
    use nalgebra::DMatrix;

    let mut mat_data = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            mat_data[j * n + i] = dtd[[i, j]] * rho + if i == j { 1.0 } else { 0.0 };
        }
    }
    let na_mat = DMatrix::from_vec(n, n, mat_data);
    let inv = na_mat
        .try_inverse()
        .ok_or_else(|| RustyStatsError::LinearAlgebraError(
            "inv(I + ρ DᵀD) falló — matriz singular".to_string()
        ))?;

    let mut result = ndarray::Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            result[[i, j]] = inv[(i, j)];
        }
    }
    Ok(result)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::regularization::smurf_matrices::{
        build_fused_lasso_matrix, eigendecompose_penalty,
    };
    use ndarray::array;

    #[test]
    fn test_proximal_lasso_matches_soft_threshold() {
        let beta = array![3.0, -2.0, 0.5, -0.3, 1.0];
        let slambda = array![1.0, 1.0, 1.0, 1.0, 1.0];
        let result = proximal_lasso(&beta, &slambda);
        assert!((result[0] - 2.0).abs() < 1e-10);
        assert!((result[1] - (-1.0)).abs() < 1e-10);
        assert!((result[2] - 0.0).abs() < 1e-10); // shrunk to zero
        assert!((result[3] - 0.0).abs() < 1e-10); // shrunk to zero
        assert!((result[4] - 0.0).abs() < 1e-10); // shrunk to zero
    }

    #[test]
    fn test_proximal_lasso_variable_threshold() {
        let beta = array![5.0, 5.0];
        let slambda = array![2.0, 4.0];
        let result = proximal_lasso(&beta, &slambda);
        assert!((result[0] - 3.0).abs() < 1e-10);
        assert!((result[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_proximal_group_lasso_shrinks() {
        let beta = array![3.0, 4.0]; // norma = 5
        let result = proximal_group_lasso(&beta, 2.0, 5.0);
        // scale = max(1 - 2/5, 0) = 0.6
        assert!((result[0] - 1.8).abs() < 1e-10);
        assert!((result[1] - 2.4).abs() < 1e-10);
    }

    #[test]
    fn test_proximal_group_lasso_zero_norm() {
        let beta = array![0.0, 0.0, 0.0];
        let result = proximal_group_lasso(&beta, 1.0, 0.0);
        assert!(result.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_proximal_group_lasso_large_slambda() {
        // slambda > norm → resultado cero
        let beta = array![1.0, 1.0];
        let norm = 2.0_f64.sqrt();
        let result = proximal_group_lasso(&beta, norm + 1.0, norm);
        assert!(result.iter().all(|&v| v.abs() < 1e-10));
    }

    #[test]
    fn test_admm_fused_lasso_converges() {
        // FusedLasso con 5 categorías en cadena
        // Solución: las categorías deben quedar fusionadas o suavizadas
        let d = build_fused_lasso_matrix(4, None).unwrap();
        let eigen = eigendecompose_penalty(&d).unwrap();

        let beta_tilde = array![1.0, 3.1, 3.0, 5.0];  // n=4 (ncols de D)
        let beta_old   = array![0.0, 0.0, 0.0, 0.0];
        let pen_weights = Array1::ones(d.nrows());
        let lambda1 = Array1::zeros(4);

        let result = proximal_admm_fused(
            &beta_tilde,
            &beta_old,
            &d,
            &eigen,
            &pen_weights,
            0.5,  // slambda
            &lambda1,
            0.0,  // lambda2
            10_000,
            1.0,
        ).unwrap();

        // El resultado debe ser finito y tener la longitud correcta
        assert_eq!(result.len(), 4);
        assert!(result.iter().all(|v| v.is_finite()));

        // Con slambda razonable, las categorías similares deben acercarse
        // (3.0, 3.1, 3.0 deben fusionarse hacia un valor común)
        // Con slambda razonable, categorías similares (3.1 y 3.0) deben acercarse
        let diff_12 = (result[1] - result[2]).abs();
        assert!(diff_12 < (3.1 - 3.0), "esperaba fusión: diff={}", diff_12);
    }

    #[test]
    fn test_admm_no_penalty_identity() {
        // Con slambda=0, el ADMM debe devolver beta_tilde sin cambios
        let d = build_fused_lasso_matrix(3, None).unwrap();
        let eigen = eigendecompose_penalty(&d).unwrap();

        let beta_tilde = array![1.0, 2.0, 3.0];  // n=3 (ncols de D)
        let beta_old   = array![0.0, 0.0, 0.0];
        let pen_weights = Array1::ones(d.nrows());
        let lambda1 = Array1::zeros(3);

        let result = proximal_admm_fused(
            &beta_tilde, &beta_old, &d, &eigen,
            &pen_weights, 0.0, &lambda1, 0.0, 10_000, 1.0,
        ).unwrap();

        // Con penalización cero, debe ser muy cercano a beta_tilde
        assert_eq!(result.len(), 3);
        for (r, b) in result.iter().zip(beta_tilde.iter()) {
            assert!((r - b).abs() < 1e-4, "diff={}", (r - b).abs());
        }
    }

    #[test]
    fn test_apply_proximal_operator_none() {
        let beta = array![1.0, 2.0, 3.0];
        let pw = Array1::ones(3);
        let l1 = Array1::zeros(3);
        let result = apply_proximal_operator(
            &beta, &beta, &SmurfPenaltyType::None,
            &pw, None, None, 0.1, 1.0, None, &l1, 0.0,
        ).unwrap();
        assert_eq!(result, beta);
    }

    #[test]
    fn test_apply_proximal_operator_lasso() {
        let beta = array![3.0, -2.0, 0.5];
        let pw = Array1::ones(3);
        let l1 = Array1::zeros(3);
        // lambda=1, step=1 → slambda=1 por coef
        let result = apply_proximal_operator(
            &beta, &beta, &SmurfPenaltyType::Lasso,
            &pw, None, None, 1.0, 1.0, None, &l1, 0.0,
        ).unwrap();
        assert!((result[0] - 2.0).abs() < 1e-10);
        assert!((result[1] - (-1.0)).abs() < 1e-10);
        assert!((result[2] - 0.0).abs() < 1e-10);
    }
}
