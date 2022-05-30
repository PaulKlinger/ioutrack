use numpy::pyo3::prelude::*;

mod ndarray_utils;
mod num_utils;

pub mod bbox;
pub mod box_tracker;
pub mod kalman;
pub mod trackers;

pub use box_tracker::KalmanBoxTracker;
pub use trackers::BaseTracker;
pub use trackers::ByteTrack;
pub use trackers::Sort;

/// A Python module implemented in Rust.
#[pymodule]
fn ioutrack(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<BaseTracker>()?;
    m.add_class::<Sort>()?;
    m.add_class::<ByteTrack>()?;
    m.add_class::<KalmanBoxTracker>()?;

    Ok(())
}
