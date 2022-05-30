use numpy::pyo3::exceptions::PyNotImplementedError;
use numpy::pyo3::prelude::*;
use numpy::PyArray2;

#[pyclass(subclass)]
pub struct BaseTracker {}

#[pymethods]
impl BaseTracker {
    #[new]
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        BaseTracker {}
    }

    #[args(boxes, return_all = "false")]
    #[pyo3(name = "update", text_signature = "(boxes, return_all = False)")]
    fn py_update<'py>(
        &mut self,
        _py: Python<'py>,
        _boxes: &'py PyAny,
        _return_all: bool,
    ) -> PyResult<&'py PyArray2<f32>> {
        Err(PyNotImplementedError::new_err(
            "Abstract method cannot be called!",
        ))
    }

    #[args(return_all = "false")]
    #[pyo3(
        name = "get_current_track_boxes",
        text_signature = "(return_all = False)"
    )]
    pub fn get_current_track_boxes<'py>(
        &self,
        _py: Python<'py>,
        _return_all: bool,
    ) -> PyResult<&'py PyArray2<f32>> {
        Err(PyNotImplementedError::new_err(
            "Abstract method cannot be called!",
        ))
    }

    #[pyo3(name = "clear_trackers", text_signature = "()")]
    pub fn clear_trackers(&self) -> PyResult<()> {
        Err(PyNotImplementedError::new_err(
            "Abstract method cannot be called!",
        ))
    }

    #[args(track_id)]
    #[pyo3(name = "remove_tracker", text_signature = "(track_id)")]
    pub fn remove_tracker(&mut self, _track_id: u32) -> PyResult<()> {
        Err(PyNotImplementedError::new_err(
            "Abstract method cannot be called!",
        ))
    }
}
