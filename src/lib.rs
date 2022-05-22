use numpy::pyo3::prelude::*;

mod kalman;
mod ndarray_utils;
mod num_utils;

pub mod bbox;
pub mod box_tracker;
pub mod trackers;

use box_tracker::KalmanBoxTracker;
pub use trackers::ByteTrack;
pub use trackers::SORTTracker;

/// A Python module implemented in Rust.
#[pymodule]
fn ioutrack(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<SORTTracker>()?;
    m.add_class::<ByteTrack>()?;
    m.add_class::<KalmanBoxTracker>()?;

    Ok(())
}
