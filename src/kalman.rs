use crate::ndarray_utils::invert_ndmatrix;
use anyhow::{Context, Result};
use nalgebra::RealField;
use ndarray::prelude::*;
use ndarray::LinalgScalar;
use num::Float;

/// Parameters to initialise a KalmanFilter
pub struct KalmanFilterParams<T: LinalgScalar + RealField + Float> {
    /// dimension of state vector
    pub dim_x: usize,
    /// dimension of measurement vectors
    pub dim_z: usize,
    /// initial mean of state estimate
    /// shape = (dim_x)
    pub x: Array1<T>,
    /// initial state covariance matrix
    /// shape = (dim_x, dim_x)
    pub p: Array2<T>,
    /// state transition matrix
    /// This is multiplied with the current state to
    /// get the prediction of the new state
    /// shape = (dim_x, dim_x)
    pub f: Array2<T>,
    /// measurement function
    /// multiplying this with the state gives the
    /// expected measurement in that state
    /// shape = (dim_z, dim_x)
    pub h: Array2<T>,
    /// measurement noise covariance matrix
    /// i.e. how much uncertainty is in the measurements
    /// shape = (dim_z, dim_z)
    pub r: Array2<T>,
    /// process noise
    /// i.e. how much uncertainty is in the transition from
    /// one state to the next
    /// shape = (dim_x, dim_x)
    pub q: Array2<T>,
}

/// Linear Kalman filter
/// use KalmanFilter::new to initialise
#[derive(Debug, Clone)]
pub struct KalmanFilter<T: LinalgScalar + RealField + Float> {
    pub x: Array2<T>,
    pub p: Array2<T>,
    f: Array2<T>,
    h: Array2<T>,
    r: Array2<T>,
    q: Array2<T>,
    y: Array2<T>,
    k: Array2<T>,
    s: Array2<T>,
    si: Array2<T>,
    _i: Array2<T>,
}

impl<T: LinalgScalar + RealField + Float> KalmanFilter<T> {
    /// Initialise Kalman filter with the given parameters
    pub fn new(params: KalmanFilterParams<T>) -> Self {
        debug_assert_eq!(
            params.x.shape(),
            &[params.dim_x],
            "Shape of x must be (dim_x)!"
        );
        debug_assert_eq!(
            params.p.shape(),
            &[params.dim_x, params.dim_x],
            "Shape of p must be (dim_x, dim_x)!"
        );
        debug_assert_eq!(
            params.f.shape(),
            &[params.dim_x, params.dim_x],
            "Shape of f must be (dim_x, dim_x)!"
        );
        debug_assert_eq!(
            params.h.shape(),
            &[params.dim_z, params.dim_x],
            "Shape of h must be (dim_z, dim_x)!"
        );
        debug_assert_eq!(
            params.r.shape(),
            &[params.dim_z, params.dim_z],
            "Shape of r must be (dim_z, dim_z)!"
        );
        debug_assert_eq!(
            params.q.shape(),
            &[params.dim_x, params.dim_x],
            "Shape of q must be (dim_x, dim_x)!"
        );

        KalmanFilter {
            x: params.x.insert_axis(Axis(1)),
            p: params.p,
            f: params.f,
            h: params.h,
            r: params.r,
            q: params.q,
            y: Array2::zeros((params.dim_z, 1)),
            k: Array2::zeros((params.dim_x, params.dim_z)),
            s: Array2::zeros((params.dim_z, params.dim_z)),
            si: Array2::zeros((params.dim_z, params.dim_z)),
            _i: Array2::eye(params.dim_x),
        }
    }

    /// Predict next state and return mean of predicted distribution
    /// check KalmanFilter.p if you need the state covariance
    /// result shape = (dim_x)
    pub fn predict(&mut self) -> ArrayView1<T> {
        self.x = self.f.dot(&self.x);
        self.p = self.f.dot(&self.p).dot(&self.f.t()) + &self.q;

        self.x.slice(s![.., 0])
    }

    /// Update estimate of the state given the measurement z and return mean
    /// check KalmanFilter.p if you need the state covariance
    /// result shape = (dim_x)
    pub fn update(&mut self, z: CowArray<T, Ix1>) -> Result<Array1<T>> {
        let z2 = z.into_owned().insert_axis(Axis(1));
        self.y = z2 - self.h.dot(&self.x);
        let pht = self.p.dot(&self.h.t());
        self.s = self.h.dot(&pht) + &self.r;
        self.si = invert_ndmatrix(self.s.view()).context("Error inverting S matrix!")?;

        self.k = pht.dot(&self.si);
        self.x += &self.k.dot(&self.y);
        let ikh = &self._i - self.k.dot(&self.h);
        self.p = ikh.dot(&self.p).dot(&ikh.t()) + self.k.dot(&self.r).dot(&self.k.t());

        // We clone the x, as the caller might do anything with it
        Ok(self.x.slice(s![.., 0]).to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_kalman_const_vel() {
        // constant velocity model
        let mut kf = KalmanFilter::<f64>::new(KalmanFilterParams {
            dim_x: 2,
            dim_z: 1,
            x: array![0., 0.],
            p: array![[0.1, 0.], [0., 10.]],
            f: array![[1., 1.], [0., 1.]],
            h: array![[1., 0.]],
            r: array![[0.5]],
            q: array![[0.1, 0.], [0., 0.2]],
        });

        assert_eq!(kf.predict(), array![0., 0.]); // state mean remains at [0, 0]
        kf.update(array![2.].into()).unwrap();
        kf.predict();
        kf.update(array![3.5].into()).unwrap();

        assert_abs_diff_eq!(kf.predict(), array![5.290117, 1.742009], epsilon = 0.0001);
    }
}
