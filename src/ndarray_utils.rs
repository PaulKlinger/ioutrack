use anyhow;

use ndarray::prelude::*;
use ndarray::Zip;
use num::Float;
use itertools::Itertools;
use itertools::EitherOrBoth::{Both, Left, Right};

fn get_output_shape(shape_a: &[usize], shape_b: &[usize]) -> anyhow::Result<Vec<usize>> {
    let mut shape = Vec::new();
    for pair in shape_a.iter().rev().zip_longest(shape_b.iter().rev()) {
        match pair {
            Both(&a_length, &b_length) => {
                if (a_length != b_length) && (a_length != 1) && (b_length != 1) {
                    return Err(anyhow::anyhow!("Inputs must have same shape or be broadcastable!"))
                }
                shape.push(if a_length > 1 {a_length} else {b_length});
            },
            Left(&l) => {shape.push(l)},
            Right(&l) => {shape.push(l)},
        }
    }
    shape.reverse();
    Ok(shape)
}

pub fn maximum<T: 'static + Float, D1: Dimension, D2: Dimension>(a: ArrayView<T, D1>, b: ArrayView<T, D2>) -> anyhow::Result<Array<T, IxDyn>>{
    let mut res = Array::zeros(get_output_shape(a.shape(), b.shape())?);
    Zip::from(&mut res).and_broadcast(&a).and_broadcast(&b).for_each(|r, &x, &y| {*r = x.max(y)});
    Ok(res)
}

pub fn minimum<T: 'static + Float, D1: Dimension, D2: Dimension>(a: ArrayView<T, D1>, b: ArrayView<T, D2>) -> anyhow::Result<Array<T, IxDyn>>{
    let mut res = Array::zeros(get_output_shape(a.shape(), b.shape())?);
    Zip::from(&mut res).and_broadcast(&a).and_broadcast(&b).for_each(|r, &x, &y| {*r = x.min(y)});
    Ok(res)
}
