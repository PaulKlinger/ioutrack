use criterion::{criterion_group, criterion_main, Criterion};
use ioutrack::bbox::ious;
use ndarray::prelude::*;
use ndarray_npy::read_npy;

fn calc_ious(boxes1: ArrayView2<f32>, boxes2: ArrayView2<f32>) -> Array2<f32> {
    ious(boxes1, boxes2)
}

pub fn criterion_bbox_benchmark(c: &mut Criterion) {
    let dets: Array2<f32> = read_npy("benches/data/mot_20-03_yolox_500_dets.npy").unwrap();
    let boxes1 = dets.slice(s![0..200, 0..4]);
    let boxes2 = dets.slice(s![200..400, 0..4]);
    c.bench_function("calc_ious_200_200", |b| {
        b.iter(|| calc_ious(boxes1, boxes2))
    });
}

criterion_group! {
    name=bbox_benches;
    config = Criterion::default();
    targets = criterion_bbox_benchmark
}
criterion_main!(bbox_benches);
