use ndarray::prelude::*;
use ndarray::LinalgScalar;
use ndarray_linalg::solve::Inverse;
use ndarray_linalg::types::Lapack;
use num::Float;
use anyhow::{Result, Context};


pub struct KalmanFilterParams<T: LinalgScalar + Lapack + Float> {
    pub dim_x: usize,
    pub dim_z: usize,
    pub f: Array2<T>,
    pub h: Array2<T>,
    pub r: Array2<T>,
    pub p: Array2<T>,
    pub q: Array2<T>,
    pub x: Array1<T>,
}

pub struct KalmanFilter<T: LinalgScalar + Lapack + Float> {
    x: Array2<T>,
    p: Array2<T>,
    q: Array2<T>,
    r: Array2<T>,
    f: Array2<T>,
    h: Array2<T>,
    y: Array2<T>,
    k: Array2<T>,
    s: Array2<T>,
    si: Array2<T>,
    _i: Array2<T>,
}


impl<T: LinalgScalar + Lapack + Float> KalmanFilter<T> {
    pub fn new(params: KalmanFilterParams<T>) -> Self {

        KalmanFilter {
            x: params.x.insert_axis(Axis(1)),
            p: params.p, q: params.q,
            r: params.r,
            f: params.f, h: params.h,
            y: Array2::zeros((params.dim_z, 1)),
            k: Array2::zeros((params.dim_x, params.dim_z)),
            s: Array2::zeros((params.dim_z, params.dim_z)),
            si: Array2::zeros((params.dim_z, params.dim_z)),
            _i: Array2::eye(params.dim_x),
        }
    }

    pub fn predict(&mut self) -> Array1<T>{
        self.x = self.f.dot(&self.x);
        self.p = self.f.dot(&self.p).dot(&self.f.t()) + &self.q;

        // We clone the x, as the caller might do anything with it
        self.x.slice(s![..,0]).to_owned()
    }

    pub fn update(&mut self, z: Array1<T>) -> Result<()> {
        let z2 = z.insert_axis(Axis(1));
        self.y = z2 - self.h.dot(&self.x);
        let pht = self.p.dot(&self.h.t());
        self.s = self.h.dot(&pht) + &self.r;
        self.si = self.s.inv().context("Error inverting S matrix!")?;

        self.k = pht.dot(&self.si);
        self.x += &self.k.dot(&self.y);
        let ikh = &self._i - self.k.dot(&self.h);
        self.p = ikh.dot(&self.p).dot(&ikh.t()) + self.k.dot(&self.r).dot(&self.k.t());
        Ok(())
    }
}