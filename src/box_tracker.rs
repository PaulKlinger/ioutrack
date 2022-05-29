use crate::bbox::Bbox;
use crate::kalman::{KalmanFilter, KalmanFilterParams};
use anyhow::Result;
use ndarray::prelude::*;
use pyo3::prelude::*;

#[pyclass]
#[derive(Debug, Clone)]
pub struct KalmanBoxTracker {
    /// track id
    #[pyo3(get)]
    pub id: u32,
    /// Kalman filter tracking bbox state
    kf: KalmanFilter<f32>,
    /// number of steps tracker has been run for (each predict() is one step)
    #[pyo3(get)]
    pub age: u32,
    /// number of steps with matching detection box
    #[pyo3(get)]
    pub hits: u32,
    /// number of consecutive steps with matched box
    #[pyo3(get)]
    pub hit_streak: u32,
    /// number of consecutive steps predicted without receiving box
    #[pyo3(get)]
    pub steps_since_update: u32,
}

pub struct KalmanBoxTrackerParams {
    pub id: u32,
    pub bbox: Bbox<f32>,
    /// Diagonal of the measurement noise covariance matrix
    /// i.e. uncertainties of (x, y, s, r) measurements
    /// default = [1., 1., 10., 10.]
    pub meas_var: Option<[f32; 4]>,
    /// Diagonal of the process noise covariance matrix
    /// i.e. uncertainties of (x, y, s, r, dx, dy, ds) during transition
    /// default = [1., 1., 1., 1. 0.01, 0.01, 0.0001]
    pub proc_var: Option<[f32; 7]>,
}

impl KalmanBoxTracker {
    /// Create new Kalman filter-based bbox tracker
    pub fn new(p: KalmanBoxTrackerParams) -> Self {
        KalmanBoxTracker {
            id: p.id,
            kf: KalmanFilter::new(KalmanFilterParams {
                dim_x: 7, // center_x, center_y, area, aspect_ratio, vel_x, vel_y, vel_area
                dim_z: 4, // center_x, center_y, area, aspect_ratio
                x: ndarray::concatenate![Axis(0), p.bbox.to_z(), array![0., 0., 0.]],
                p: Array2::from_diag(&array![10., 10., 10., 10., 10000., 10000., 10000.]),
                f: array![
                    [1., 0., 0., 0., 1., 0., 0.], // center_x' = center_x + vel_x
                    [0., 1., 0., 0., 0., 1., 0.], // center_y' = center_y + vel_y
                    [0., 0., 1., 0., 0., 0., 1.], // area' = area + vel_area
                    [0., 0., 0., 1., 0., 0., 0.], // aspect_ratio' = aspect_ratio
                    [0., 0., 0., 0., 1., 0., 0.], // vel_x' = vel_x
                    [0., 0., 0., 0., 0., 1., 0.], // vel_y' = vel_y
                    [0., 0., 0., 0., 0., 0., 1.], // vel_area' = vel_area
                ],
                h: array![
                    [1., 0., 0., 0., 0., 0., 0.],
                    [0., 1., 0., 0., 0., 0., 0.],
                    [0., 0., 1., 0., 0., 0., 0.],
                    [0., 0., 0., 1., 0., 0., 0.]
                ],
                r: Array2::from_diag(&arr1(&p.meas_var.unwrap_or([1., 1., 10., 10.]))),
                q: Array2::from_diag(&arr1(
                    &p.proc_var.unwrap_or([1., 1., 1., 1., 0.01, 0.01, 0.0001]),
                )),
            }),
            age: 0,
            hits: 0,
            hit_streak: 0,
            steps_since_update: 0,
        }
    }

    /// Update tracker with detected box
    pub fn update(&mut self, bbox: Bbox<f32>) -> Result<()> {
        // don't increase hits/hit_streak if we get
        // several updates in the same step
        if self.steps_since_update > 0 {
            self.hits += 1;
            self.hit_streak += 1;
        }
        self.steps_since_update = 0;
        self.kf.update(bbox.to_z().into())?;
        Ok(())
    }

    /// Predict box position in next step
    pub fn predict(&mut self) -> Bbox<f32> {
        self.age += 1;
        if self.steps_since_update > 0 {
            self.hit_streak = 0;
        }
        self.steps_since_update += 1;
        // next predict would cause area to be non-positive...
        if self.kf.x[[2, 0]] + self.kf.x[[6, 0]] <= 0. {
            // ...so set vel_area to zero
            self.kf.x[[6, 0]] = 0.;
        }

        let pred_x = self.kf.predict();
        Bbox::from_z(pred_x.slice(s![0..4])).unwrap()
    }

    pub fn bbox(&self) -> Bbox<f32> {
        Bbox::from_z(self.kf.x.slice(s![0..4, 0])).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_bbox_tracker() {
        let mut tracker = KalmanBoxTracker::new(KalmanBoxTrackerParams {
            id: 0,
            bbox: Bbox {
                xmin: 0.,
                xmax: 10.,
                ymin: 0.,
                ymax: 5.,
            },
            meas_var: None,
            proc_var: None,
        });
        tracker.predict();
        tracker
            .update(Bbox {
                xmin: 5.,
                xmax: 15.,
                ymin: 0.,
                ymax: 4.5,
            })
            .unwrap();

        let pred = tracker.predict();

        assert_abs_diff_eq!(
            pred,
            Bbox {
                xmin: 10.392181,
                xmax: 19.594833,
                ymin: -0.173802,
                ymax: 4.1744514
            },
            epsilon = 0.0001
        );
    }
}
