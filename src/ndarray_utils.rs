use crate::num_utils::{partial_max, partial_min, BboxNum};
use anyhow::Result;
use itertools::EitherOrBoth::{Both, Left, Right};
use itertools::Itertools;
use nalgebra::{try_invert_to, DMatrix};
use nalgebra::{RealField, Scalar};
use ndarray::prelude::*;
use ndarray::Zip;
use nshare::ToNalgebra;
use nshare::ToNdarray2;

/// Get the output shape when broadcasting a & b against each other
/// (Can give output that's bigger than a or b in each axis, vs std rust ndarray broadcasting
///  which only expands the second argument.)
/// e.g. [3, 2, 1], [2, 5] -> [3, 2, 5]
fn get_broadcast_shape(shape_a: &[usize], shape_b: &[usize]) -> anyhow::Result<Vec<usize>> {
    let mut shape = Vec::new();
    for pair in shape_a.iter().rev().zip_longest(shape_b.iter().rev()) {
        match pair {
            Both(&a_length, &b_length) => {
                if (a_length != b_length) && (a_length != 1) && (b_length != 1) {
                    return Err(anyhow::anyhow!(
                        "Inputs must have same shape or be broadcastable!"
                    ));
                }
                shape.push(if a_length != 1 { a_length } else { b_length });
            }
            Left(&l) => shape.push(l),
            Right(&l) => shape.push(l),
        }
    }
    shape.reverse();
    Ok(shape)
}

/// Elementwise maximum of two arrays, with numpy-style broadcasting
pub fn maximum<T: BboxNum, D1: Dimension, D2: Dimension>(
    a: ArrayView<T, D1>,
    b: ArrayView<T, D2>,
) -> anyhow::Result<Array<T, IxDyn>> {
    let mut res = Array::zeros(get_broadcast_shape(a.shape(), b.shape())?);
    Zip::from(&mut res)
        .and_broadcast(&a)
        .and_broadcast(&b)
        .for_each(|r, &x, &y| *r = partial_max(x, y));
    Ok(res)
}

/// Elementwise minimum of two arrays, with numpy-style broadcasting
pub fn minimum<T: BboxNum, D1: Dimension, D2: Dimension>(
    a: ArrayView<T, D1>,
    b: ArrayView<T, D2>,
) -> anyhow::Result<Array<T, IxDyn>> {
    let mut res = Array::zeros(get_broadcast_shape(a.shape(), b.shape())?);
    Zip::from(&mut res)
        .and_broadcast(&a)
        .and_broadcast(&b)
        .for_each(|r, &x, &y| *r = partial_min(x, y));
    Ok(res)
}

/// Invert the given square matrix if possible
pub fn invert_ndmatrix<T: RealField + Scalar>(a: ArrayView2<T>) -> Result<Array2<T>> {
    let nalg_array = a.into_nalgebra().into_owned();
    let (rows, cols) = a.dim();
    let mut out = DMatrix::from_element(rows, cols, T::zero());
    let res = try_invert_to(nalg_array, &mut out);
    match res {
        true => Ok(out.into_ndarray2()),
        false => Err(anyhow::anyhow!("Error inverting matrix!")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_get_broadcast_shape_successful() {
        assert_eq!(get_broadcast_shape(&[3, 2, 1], &[2, 5]).unwrap(), [3, 2, 5]);
        assert_eq!(get_broadcast_shape(&[3, 3], &[3, 3]).unwrap(), [3, 3]);
        assert_eq!(get_broadcast_shape(&[1, 1], &[5, 5]).unwrap(), [5, 5]);
        assert_eq!(get_broadcast_shape(&[3, 10, 3], &[3]).unwrap(), [3, 10, 3]);
    }

    #[test]
    #[should_panic]
    fn test_get_broadcast_shape_fail() {
        get_broadcast_shape(&[3, 2, 2], &[2, 5]).unwrap();
    }

    #[test]
    fn test_maximum() {
        let a: Array2<i64> = array![[1, 5, 1]];
        let b = array![[3], [-1]];
        let max = maximum(a.view(), b.view())
            .unwrap()
            .into_dimensionality()
            .unwrap();
        assert_eq!(max, array![[3, 5, 3], [1, 5, 1]]);
    }

    #[test]
    fn test_minimum() {
        let a: Array2<i64> = array![[1, 5, 1]];
        let b = array![[3], [-1]];
        let max = minimum(a.view(), b.view())
            .unwrap()
            .into_dimensionality()
            .unwrap();
        assert_eq!(max, array![[1, 3, 1], [-1, -1, -1]]);
    }

    #[test]
    fn test_invert() {
        let a: Array2<f64> = array![[2., 6.], [1., 4.]];
        let ainv = invert_ndmatrix(a.view()).unwrap();
        assert_abs_diff_eq!(ainv, array![[2., -3.], [-0.5, 1.]], epsilon = 0.0001)
    }
}
