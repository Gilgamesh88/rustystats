// =============================================================================
// Spline Basis Benchmarks
// =============================================================================
//
// Measures B-spline and natural spline basis construction performance.
// Run with: cargo bench --workspace
// =============================================================================

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::Array1;

use rustystats_core::splines::{bs, ns};

/// Generate evenly-spaced values for benchmarking.
fn generate_x(n: usize) -> Array1<f64> {
    Array1::from_vec((0..n).map(|i| i as f64 / n as f64 * 10.0).collect())
}

fn bench_bspline(c: &mut Criterion) {
    let mut group = c.benchmark_group("B-spline basis");

    for &(n, df) in &[(100, 5), (1000, 10), (5000, 20), (10000, 30)] {
        let x = generate_x(n);

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("n={n}_df={df}")),
            &(n, df),
            |b, &(_, df)| {
                b.iter(|| {
                    bs(black_box(&x), df);
                });
            },
        );
    }
    group.finish();
}

fn bench_natural_spline(c: &mut Criterion) {
    let mut group = c.benchmark_group("Natural spline basis");

    for &(n, df) in &[(100, 5), (1000, 10), (5000, 20), (10000, 30)] {
        let x = generate_x(n);

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("n={n}_df={df}")),
            &(n, df),
            |b, &(_, df)| {
                b.iter(|| {
                    ns(black_box(&x), df);
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_bspline, bench_natural_spline);
criterion_main!(benches);
