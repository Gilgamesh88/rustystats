// =============================================================================
// SMuRF — Matrices de penalización D (faer sparse)
// =============================================================================
//
// Cada función construye la matriz D para un tipo de penalización.
// El operador proximal ADMM usa D en el hot-path — por eso sparse es crítico
// a 500+ nodos (250x menos memoria que densa).
//
// Porta desde R:
//   build_lasso_matrix()           ← .pen.mat.lasso()
//   build_group_lasso_matrix()     ← .pen.mat.grouplasso()
//   build_fused_lasso_matrix()     ← .pen.mat.flasso()
//   build_gen_fused_lasso_matrix() ← .pen.mat.gflasso()
//   build_graph_guided_matrix()    ← .pen.mat.ggflasso()  [principal, sparse]
//   build_2d_fused_lasso_matrix()  ← .pen.mat.2dflasso()
//   eigendecompose_penalty()       ← .pen.mat.eig()
// =============================================================================

use faer::sparse::SparseColMat;
use faer::Mat;
use ndarray::Array2;

use crate::error::{Result, RustyStatsError};

// =============================================================================
// Tipos de salida
// =============================================================================

/// Datos de eigendecomposición de DᵀD, precalculados antes del loop FISTA.
/// Se pasan al ADMM para evitar recomputar en cada iteración.
pub struct EigenData {
    /// Matriz de eigenvectores Q (columnas = eigenvectores de DᵀD).
    pub q: Mat<f64>,
    /// Vector de eigenvalores de DᵀD.
    pub eigvals: Vec<f64>,
    /// True si todos los eigenvalores son > ε (habilita el path rápido del ADMM).
    pub fast_mode: bool,
}

/// Matriz de penalización D en formato sparse (faer).
/// Dimensión: (n_constraints × n_params).
pub type PenaltyMatrix = SparseColMat<usize, f64>;

// =============================================================================
// Helper interno: construir SparseColMat desde triplets (row, col, val)
// =============================================================================

fn sparse_from_triplets(
    nrows: usize,
    ncols: usize,
    triplets: &[(usize, usize, f64)],
) -> Result<PenaltyMatrix> {
    let faer_triplets: Vec<faer::sparse::Triplet<usize, usize, f64>> = triplets
        .iter()
        .map(|&(r, c, v)| faer::sparse::Triplet { row: r, col: c, val: v })
        .collect();

    SparseColMat::<usize, f64>::try_new_from_triplets(nrows, ncols, &faer_triplets)
        .map_err(|e| RustyStatsError::LinearAlgebraError(format!("sparse_from_triplets: {:?}", e)))
}

// =============================================================================
// 1. Lasso — matriz identidad I_n
// =============================================================================
//
// D = I_n  (diagonal de 1s)
// Dimensión: n × n
// No-ceros: n (uno por fila)
//
// El PO de Lasso no necesita realmente D — usa soft_threshold directo.
// Esta función existe para uniformidad de la interfaz.

pub fn build_lasso_matrix(n: usize) -> Result<PenaltyMatrix> {
    let triplets: Vec<(usize, usize, f64)> = (0..n).map(|i| (i, i, 1.0)).collect();
    sparse_from_triplets(n, n, &triplets)
}

// =============================================================================
// 2. Group Lasso — matriz identidad (igual que Lasso)
// =============================================================================

pub fn build_group_lasso_matrix(n: usize) -> Result<PenaltyMatrix> {
    build_lasso_matrix(n)
}

// =============================================================================
// 3. Fused Lasso — diferencias entre categorías consecutivas
// =============================================================================
//
// Para n categorías (con categoría de referencia = primera):
//   D tiene dimensión (n-1) × n
//   Fila i: -1 en col i, +1 en col i+1
//
// Sin categoría de referencia (ref_cat = None):
//   D tiene dimensión n × (n+1), luego se elimina la fila ref_cat
//
// Equivale a .pen.mat.flasso(n.par, refcat) en R.

pub fn build_fused_lasso_matrix(n_params: usize, ref_cat: Option<usize>) -> Result<PenaltyMatrix> {
    // n_params = número de parámetros a estimar (sin contar ref_cat)
    // n_levels = n_params + 1 si hay ref_cat, o n_params si no
    let n_levels = match ref_cat {
        Some(_) => n_params + 1,
        None    => n_params,
    };

    if n_levels < 2 {
        return Err(RustyStatsError::InvalidValue(
            "FusedLasso necesita al menos 2 niveles".to_string()
        ));
    }

    // Construir diferencias consecutivas sobre todos los niveles
    // Fila i: col i → -1, col i+1 → +1   (i = 0..n_levels-2)
    let n_rows_full = n_levels - 1;
    let mut triplets: Vec<(usize, usize, f64)> = Vec::with_capacity(n_rows_full * 2);

    for i in 0..n_rows_full {
        triplets.push((i, i,     -1.0));
        triplets.push((i, i + 1,  1.0));
    }

    // Matriz completa: n_rows_full × n_levels
    let d_full = sparse_from_triplets(n_rows_full, n_levels, &triplets)?;

    // Si hay categoría de referencia, eliminar la columna correspondiente
    // (equivalente a eliminar la fila ref_cat en R, que opera en el transpuesto)
    match ref_cat {
        None => Ok(d_full),
        Some(rc) => remove_column(&d_full, rc, n_rows_full, n_levels),
    }
}

// =============================================================================
// 4. Generalized Fused Lasso — todas las diferencias por pares
// =============================================================================
//
// D tiene dimensión C(n,2) × n  (todas las combinaciones de pares)
// Fila para par (i,j) con i<j: -1 en col i, +1 en col j
//
// Equivale a .pen.mat.gflasso(n.par, refcat) en R.

pub fn build_gen_fused_lasso_matrix(n_params: usize, ref_cat: Option<usize>) -> Result<PenaltyMatrix> {
    let n_levels = match ref_cat {
        Some(_) => n_params + 1,
        None    => n_params,
    };

    if n_levels < 2 {
        return Err(RustyStatsError::InvalidValue(
            "GenFusedLasso necesita al menos 2 niveles".to_string()
        ));
    }

    let n_pairs = n_levels * (n_levels - 1) / 2;
    let mut triplets: Vec<(usize, usize, f64)> = Vec::with_capacity(n_pairs * 2);
    let mut row = 0usize;

    for i in 0..n_levels {
        for j in (i + 1)..n_levels {
            triplets.push((row, i, -1.0));
            triplets.push((row, j,  1.0));
            row += 1;
        }
    }

    let d_full = sparse_from_triplets(n_pairs, n_levels, &triplets)?;

    match ref_cat {
        None => Ok(d_full),
        Some(rc) => remove_column(&d_full, rc, n_pairs, n_levels),
    }
}

// =============================================================================
// 5. Graph-Guided Fused Lasso — incidencia del grafo de adyacencia
// =============================================================================
//
// Para cada arista (i, j) con i < j en el grafo:
//   Fila de D: +1 en col j, -1 en col i  (excluye col ref_cat si existe)
//
// D es E × n donde E = número de aristas.
// Con 500 nodos y densidad 2%, D es ~2500 × 500 con solo 2 no-ceros por fila.
//
// Equivale a .pen.mat.ggflasso(adj.matrix, lev.names, refcat) en R.

pub fn build_graph_guided_matrix(
    adj: &Array2<f64>,
    ref_cat: Option<usize>,
) -> Result<PenaltyMatrix> {
    let (n, _) = adj.dim();

    // Contar aristas (triángulo superior)
    let mut edges: Vec<(usize, usize)> = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            if adj[[i, j]] > 0.5 {
                edges.push((i, j));
            }
        }
    }

    if edges.is_empty() {
        return Err(RustyStatsError::InvalidValue(
            "adj_matrix no tiene aristas (todos los valores son 0)".to_string()
        ));
    }

    let n_edges = edges.len();

    // Columnas a incluir (todas excepto ref_cat)
    let col_map: Vec<usize> = (0..n)
        .filter(|&c| Some(c) != ref_cat)
        .collect();
    let n_cols = col_map.len();

    // Mapeo inverso: nivel original → columna en D (None si es ref_cat)
    let mut level_to_col = vec![None::<usize>; n];
    for (new_col, &orig_level) in col_map.iter().enumerate() {
        level_to_col[orig_level] = Some(new_col);
    }

    let mut triplets: Vec<(usize, usize, f64)> = Vec::with_capacity(n_edges * 2);

    for (row, &(i, j)) in edges.iter().enumerate() {
        // Nodo j → +1
        if let Some(col_j) = level_to_col[j] {
            triplets.push((row, col_j,  1.0));
        }
        // Nodo i → -1
        if let Some(col_i) = level_to_col[i] {
            triplets.push((row, col_i, -1.0));
        }
        // Si i o j es ref_cat, su contribución se omite (columna eliminada)
    }

    sparse_from_triplets(n_edges, n_cols, &triplets)
}

// =============================================================================
// 6. 2D Fused Lasso — diferencias horizontales y verticales en grilla
// =============================================================================
//
// Para una grilla de n_rows × n_cols categorías:
//   Diferencias horizontales: (i,j) - (i,j+1) por cada fila
//   Diferencias verticales:   (i,j) - (i+1,j) por cada columna
//
// Equivale a .pen.mat.2dflasso(n.par.row, n.par.col) en R.

pub fn build_2d_fused_lasso_matrix(n_rows: usize, n_cols: usize) -> Result<PenaltyMatrix> {
    if n_rows < 2 || n_cols < 2 {
        return Err(RustyStatsError::InvalidValue(
            "2DFusedLasso necesita al menos 2 filas y 2 columnas".to_string()
        ));
    }

    let total_params = n_rows * n_cols;

    // Número de diferencias horizontales + verticales
    let n_horiz = n_rows * (n_cols - 1);
    let n_vert  = (n_rows - 1) * n_cols;
    let n_constraints = n_horiz + n_vert;

    let mut triplets: Vec<(usize, usize, f64)> = Vec::with_capacity(n_constraints * 2);
    let mut row = 0usize;

    // Índice en el vector de parámetros: (i, j) → i * n_cols + j
    let idx = |i: usize, j: usize| i * n_cols + j;

    // Diferencias horizontales: (i,j) vs (i,j+1)
    for i in 0..n_rows {
        for j in 0..(n_cols - 1) {
            triplets.push((row, idx(i, j),     -1.0));
            triplets.push((row, idx(i, j + 1),  1.0));
            row += 1;
        }
    }

    // Diferencias verticales: (i,j) vs (i+1,j)
    for i in 0..(n_rows - 1) {
        for j in 0..n_cols {
            triplets.push((row, idx(i, j),         -1.0));
            triplets.push((row, idx(i + 1, j),      1.0));
            row += 1;
        }
    }

    sparse_from_triplets(n_constraints, total_params, &triplets)
}

// =============================================================================
// 7. Eigendecomposición de DᵀD
// =============================================================================
//
// Calcula DᵀD (denso) y su descomposición espectral.
// Solo se llama UNA VEZ por predictor antes del loop FISTA.
//
// El path rápido del ADMM usa:
//   ADMM_aux = I - ρ Q diag(1/(eigvals + ρ)) Qᵀ
// en lugar de calcular inv(I + ρ DᵀD) en cada iteración.
//
// Equivale a .pen.mat.eig() en R.

pub fn eigendecompose_penalty(d: &PenaltyMatrix) -> Result<EigenData> {
    let (m, n) = (d.nrows(), d.ncols());

    // 1. Calcular DᵀD como matriz densa
    //    DᵀD[i,j] = Σ_k D[k,i] * D[k,j]
    let mut dtd = Mat::<f64>::zeros(n, n);

    // Iterar columnas de D sparse para construir DᵀD
    for col_j in 0..n {
        for col_i in 0..n {
            let mut val = 0.0f64;
            // Producto punto de columna i y columna j de D
            // Como D es sparse, iteramos los no-ceros
            // Usamos la representación CSC de faer
            let col_i_entries = d.row_idx_of_col_raw(col_i);
            let col_i_vals    = d.val_of_col(col_i);
            let col_j_entries = d.row_idx_of_col_raw(col_j);
            let col_j_vals    = d.val_of_col(col_j);

            // Intersección de filas no-cero (D tiene pocas entradas por columna)
            let mut pi = 0;
            let mut pj = 0;
            while pi < col_i_entries.len() && pj < col_j_entries.len() {
                match col_i_entries[pi].cmp(&col_j_entries[pj]) {
                    std::cmp::Ordering::Equal => {
                        val += col_i_vals[pi] * col_j_vals[pj];
                        pi += 1;
                        pj += 1;
                    }
                    std::cmp::Ordering::Less    => { pi += 1; }
                    std::cmp::Ordering::Greater => { pj += 1; }
                }
            }
            dtd[(col_i, col_j)] = val;
        }
    }

    // 2. Eigendecomposición simétrica de DᵀD
    let evd = dtd.self_adjoint_eigen(faer::Side::Lower)
        .map_err(|e| RustyStatsError::LinearAlgebraError(format!("eigendecomp: {:?}", e)))?;

    let s = evd.S();
    let eigvals: Vec<f64> = (0..n).map(|i| s.column_vector()[i]).collect();

    // Eigenvectores como faer::Mat
    let q = evd.U().to_owned();

    // 3. Fast mode: todos los eigenvalores > ε numérico
    let eps = f64::EPSILON.sqrt();
    let fast_mode = eigvals.iter().all(|&v: &f64| v.abs() >= eps);

    let _ = m; // silenciar warning de m no usado

    Ok(EigenData { q, eigvals, fast_mode })
}

// =============================================================================
// Helper: eliminar una columna de una SparseColMat
// =============================================================================
//
// Necesario para FusedLasso y GenFusedLasso cuando hay categoría de referencia.

fn remove_column(
    d: &PenaltyMatrix,
    col_to_remove: usize,
    nrows: usize,
    ncols: usize,
) -> Result<PenaltyMatrix> {
    if col_to_remove >= ncols {
        return Err(RustyStatsError::InvalidValue(format!(
            "ref_cat {} fuera de rango (ncols={})", col_to_remove, ncols
        )));
    }

    let new_ncols = ncols - 1;
    let mut triplets: Vec<(usize, usize, f64)> = Vec::new();

    for col in 0..ncols {
        if col == col_to_remove { continue; }
        let new_col = if col < col_to_remove { col } else { col - 1 };
        let row_indices = d.row_idx_of_col_raw(col);
        let values      = d.val_of_col(col);
        for (&row, &val) in row_indices.iter().zip(values.iter()) {
            triplets.push((row, new_col, val));
        }
    }

    sparse_from_triplets(nrows, new_ncols, &triplets)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_lasso_matrix_shape() {
        let d = build_lasso_matrix(5).unwrap();
        assert_eq!(d.nrows(), 5);
        assert_eq!(d.ncols(), 5);
        assert_eq!(d.compute_nnz(), 5); // 5 unos en la diagonal
    }

    #[test]
    fn test_fused_lasso_no_refcat() {
        // 4 niveles sin ref_cat → D es 3×4
        let d = build_fused_lasso_matrix(4, None).unwrap();
        assert_eq!(d.nrows(), 3);
        assert_eq!(d.ncols(), 4);
        assert_eq!(d.compute_nnz(), 6); // 2 no-ceros por fila
    }

    #[test]
    fn test_fused_lasso_with_refcat() {
        // 5 niveles, ref_cat=0 → n_params=4, D es 4×4
        let d = build_fused_lasso_matrix(4, Some(0)).unwrap();
        assert_eq!(d.nrows(), 4);
        assert_eq!(d.ncols(), 4);
    }

    #[test]
    fn test_gen_fused_lasso_shape() {
        // 4 niveles → C(4,2)=6 pares, D es 6×4
        let d = build_gen_fused_lasso_matrix(4, None).unwrap();
        assert_eq!(d.nrows(), 6);
        assert_eq!(d.ncols(), 4);
        assert_eq!(d.compute_nnz(), 12); // 2 no-ceros por fila
    }

    #[test]
    fn test_graph_guided_chain() {
        // Grafo cadena: 0-1-2-3 (3 aristas)
        let adj = array![
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ];
        let d = build_graph_guided_matrix(&adj, None).unwrap();
        assert_eq!(d.nrows(), 3); // 3 aristas
        assert_eq!(d.ncols(), 4); // 4 nodos
    }

    #[test]
    fn test_graph_guided_chain_with_refcat() {
        // Misma cadena, ref_cat=0 → D es 3×3 (columna 0 eliminada)
        let adj = array![
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ];
        let d = build_graph_guided_matrix(&adj, Some(0)).unwrap();
        assert_eq!(d.nrows(), 3);
        assert_eq!(d.ncols(), 3);
    }

    #[test]
    fn test_graph_guided_no_edges_error() {
        let adj = array![[0.0, 0.0], [0.0, 0.0]];
        assert!(build_graph_guided_matrix(&adj, None).is_err());
    }

    #[test]
    fn test_2d_fused_lasso_shape() {
        // Grilla 3×4: 3*(4-1)=9 horiz + (3-1)*4=8 vert = 17 constraints
        let d = build_2d_fused_lasso_matrix(3, 4).unwrap();
        assert_eq!(d.nrows(), 17);
        assert_eq!(d.ncols(), 12); // 3*4=12 parámetros
    }

    #[test]
    fn test_eigendecompose_fused_lasso() {
        // Eigendcomp de una FusedLasso 3×3 pequeña
        let d = build_fused_lasso_matrix(3, None).unwrap();
        let eigen = eigendecompose_penalty(&d).unwrap();
        assert_eq!(eigen.eigvals.len(), 3);
        // Todos los eigenvalores deben ser >= 0 (DᵀD es semidefinida positiva)
        for &v in &eigen.eigvals {
            assert!(v >= -1e-10, "eigenvalor negativo: {}", v);
        }
    }

    #[test]
    fn test_graph_guided_500_nodes_sparse() {
        // Test de stress: 500 nodos en cadena (499 aristas)
        let mut adj = Array2::<f64>::zeros((500, 500));
        for i in 0..499 {
            adj[[i, i+1]] = 1.0;
            adj[[i+1, i]] = 1.0;
        }
        let d = build_graph_guided_matrix(&adj, None).unwrap();
        assert_eq!(d.nrows(), 499);
        assert_eq!(d.ncols(), 500);
        // Solo 2 no-ceros por arista
        assert_eq!(d.compute_nnz(), 499 * 2);
    }
}
