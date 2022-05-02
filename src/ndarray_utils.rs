use anyhow;

use ndarray::prelude::*;
use num::Float;

pub fn maximum<T: 'static + Float, D: Dimension>(a: ArrayView<T, D>, b: ArrayView<T, D>) -> anyhow::Result<Array<T, D>>{
    if a.raw_dim() != b.raw_dim() {
        return Err(anyhow::anyhow!("Inputs for 'maximum' must have same shape!"))
    }
    let mut res = Array::zeros(a.raw_dim());
    azip!((r in &mut res, &a in &a, &b in &b) *r = a.max(b));
    Ok(res)
}

pub fn minimum<T: 'static + Float, D: Dimension>(a: ArrayView<T, D>, b: ArrayView<T, D>) -> anyhow::Result<Array<T, D>>{
    if a.raw_dim() != b.raw_dim() {
        return Err(anyhow::anyhow!("Inputs for 'maximum' must have same shape!"))
    }
    let mut res = Array::zeros(a.raw_dim());
    azip!((r in &mut res, &a in &a, &b in &b) *r = a.min(b));
    Ok(res)
}