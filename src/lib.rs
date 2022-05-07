use ndarray::prelude::array;
use ndarray_linalg::solve::Inverse;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyArithmeticError;
use pyo3::prelude::*;

mod num_utils;

mod kalman;
use kalman::{KalmanFilter, KalmanFilterParams};

pub mod bbox;
use bbox::ious;

mod ndarray_utils;

#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

#[pyfunction]
fn array2_sum(a: PyReadonlyArray2<f64>) -> PyResult<f64> {
    let array = a.as_array();
    Ok(array.sum())
}

#[pyfunction]
fn array2_inv<'py>(_py: Python<'py>, a: PyReadonlyArray2<f64>) -> PyResult<&'py PyArray2<f64>> {
    let array = a.as_array();
    match array.inv() {
        Ok(inv) => Ok(inv.into_pyarray(_py)),
        Err(err) => Err(PyArithmeticError::new_err(err.to_string())),
    }
}

#[pyfunction]
fn calc_ious<'py>(
    _py: Python<'py>,
    boxes1: PyReadonlyArray2<f64>,
    boxes2: PyReadonlyArray2<f64>,
) -> PyResult<&'py PyArray2<f64>> {
    Ok(ious(boxes1.as_array(), boxes2.as_array()).into_pyarray(_py))
}

#[pyfunction]
fn test_kalman(_py: Python) -> PyResult<&PyArray1<f64>> {
    let mut kf = KalmanFilter::<f64>::new(KalmanFilterParams {
        dim_x: 2,
        dim_z: 1,
        f: array![[1., 1.], [0., 1.]],
        h: array![[1., 0.]],
        r: array![[0.5]],
        p: array![[0.1, 0.], [0., 10.]],
        q: array![[0.1, 0.], [0., 0.2]],
        x: array![0., 0.],
    });

    if let Err(err) = kf.update(array![2.]) {
        return Err(PyArithmeticError::new_err(err.to_string()));
    }

    kf.predict();
    if let Err(err) = kf.update(array![3.5]) {
        return Err(PyArithmeticError::new_err(err.to_string()));
    }

    Ok(kf.predict().into_pyarray(_py))
}

/// A Python module implemented in Rust.
#[pymodule]
fn ioutrack(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(array2_sum, m)?)?;
    m.add_function(wrap_pyfunction!(array2_inv, m)?)?;
    m.add_function(wrap_pyfunction!(test_kalman, m)?)?;
    m.add_function(wrap_pyfunction!(calc_ious, m)?)?;

    Ok(())
}
