use criterion::{criterion_group, criterion_main, Criterion};
use ioutrack::kalman::{KalmanFilter, KalmanFilterParams};
use ndarray::prelude::*;

fn kalman_lin_vel_update_predict(measurements: &Vec<ArrayView1<f32>>) {
    let mut kf = KalmanFilter::<f32>::new(KalmanFilterParams {
        dim_x: 2,
        dim_z: 1,
        x: array![0., 0.],
        p: array![[0.1, 0.], [0., 10.]],
        f: array![[1., 1.], [0., 1.]],
        h: array![[1., 0.]],
        r: array![[0.5]],
        q: array![[0.1, 0.], [0., 0.2]],
    });

    for meas in measurements {
        kf.update(meas.into()).unwrap();
        kf.predict();
    }
}

pub fn criterion_kalman_benchmark(c: &mut Criterion) {
    let measurements: Vec<Array1<f32>> = (0..100_i32).map(|i| array![i as f32]).collect();
    // kf.update(x) overwrites x if it's an owned array, so we pass in a view instead
    let measurements_views = measurements.iter().map(|m| m.view()).collect();
    c.bench_function("kalman_lin_vel_update_predict", |b| {
        b.iter(|| kalman_lin_vel_update_predict(&measurements_views))
    });
}

criterion_group! {
    name=kalman_benches;
    config = Criterion::default();
    targets = criterion_kalman_benchmark
}
criterion_main!(kalman_benches);
