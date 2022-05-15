use num::cast;
use numpy::ndarray::prelude::*;
use numpy::pyo3::exceptions::PyValueError;
use numpy::pyo3::prelude::*;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};

use lapjv::lapjv;

use crate::bbox::{ious, Bbox};
use crate::box_tracker::{KalmanBoxTracker, KalmanBoxTrackerParams};

fn assign_detections_to_tracks(
    detections: ArrayView2<f32>,
    tracks: ArrayView2<f32>,
    iou_threshold: f32,
) {
    let mut det_track_ious = ious(detections, tracks);
    det_track_ious.mapv_inplace(|x| -x);
    let res = lapjv(&det_track_ious);
}

/// Create a new SORT bbox tracker
///
/// Parameters
/// ----------
/// max_age
///     maximum frames a tracklet is kept alive without matching detections
/// min_hits
///     minimum number of successive detections before a tracklet is set to alive
/// iou_threshold
///     minimum IOU to assign detection to tracklet
/// score_threshold
///     minimum score to use a detected bbox
#[pyclass(text_signature = "(max_age=1, min_hits=3, iou_threshold=0.3, score_threshold=0.5)")]
pub struct SORTTracker {
    #[pyo3(get, set)]
    pub max_age: u32,
    #[pyo3(get, set)]
    pub min_hits: u32,
    #[pyo3(get, set)]
    pub iou_threshold: f32,
    #[pyo3(get, set)]
    pub score_threshold: f32,

    /// id of next tracklet initialized
    next_track_id: u32,
    /// current tracklets
    #[pyo3(get)]
    pub tracklets: Vec<KalmanBoxTracker>,
    /// number of steps the tracker has run for
    #[pyo3(get)]
    pub n_steps: u32,
}

impl SORTTracker {
    fn get_tracklet_boxes(&mut self) -> Array2<f32> {
        let mut data = Vec::with_capacity(self.tracklets.len() * 5);
        for t in self.tracklets.iter_mut() {
            let b = t.predict();
            data.extend([b.xmin, b.ymin, b.xmax, b.ymax, cast(t.id).unwrap()]);
        }
        Array2::from_shape_vec((self.tracklets.len(), 5), data).unwrap()
    }
}

#[pymethods]
impl SORTTracker {
    #[new]
    #[args(
        max_age = "1",
        min_hits = "3",
        iou_threshold = "0.3",
        score_threshold = "0.5"
    )]
    fn py_new(max_age: u32, min_hits: u32, iou_threshold: f32, score_threshold: f32) -> Self {
        SORTTracker {
            max_age,
            min_hits,
            iou_threshold,
            score_threshold,
            next_track_id: 0,
            tracklets: vec![KalmanBoxTracker::new(KalmanBoxTrackerParams {
                id: 0,
                bbox: Bbox {
                    xmin: 0.,
                    xmax: 10.,
                    ymin: 0.,
                    ymax: 5.,
                },
                center_var: None,
                area_var: None,
                aspect_var: None,
            })],
            n_steps: 0,
        }
    }

    /// Update the tracker with new boxes and return position of current tracklets
    ///
    /// Parameters
    /// ----------
    /// boxes
    ///     array of boxes of shape (n_boxes, 5)
    ///     of the form [[xmin1, ymin1, xmax1, ymax1, score1], [xmin2,...],...]
    /// return_all
    ///     if true return all living trackers, including inactive (but not dead) ones
    ///     otherwise return only active trackers (those that got at least min_hits
    ///     matching boxes in a row)
    ///
    /// Returns
    /// -------
    ///    array of tracklet boxes with shape (n_tracks, 5)
    ///    of the form [[xmin1, ymin1, xmax1, ymax1, track_id1], [xmin2,...],...]
    #[args(boxes, return_all = "false")]
    #[pyo3(text_signature = "(boxes, return_all = False)")]
    fn update<'py>(
        &mut self,
        _py: Python<'py>,
        boxes: &'py PyAny,
        return_all: bool,
    ) -> PyResult<&'py PyArray2<f32>> {
        // We allow boxes to be either f32 (then we use it directly)
        // or f64 (then we convert to f32)
        // TODO: Can't think of a way to move this into a separate function without copying in the f32 case...
        let boxes_py32_res: PyResult<PyReadonlyArray2<'py, f32>> = boxes.extract();
        let boxes_f32_owned: Array2<f32>; // if we convert we have to allocate a new array
                                          // in either case the result is a view of an f32 array
        let detection_boxes: ArrayView2<f32> = match boxes_py32_res {
            Ok(ref arr) => arr.as_array(),
            Err(_) => {
                boxes_f32_owned = match boxes.extract::<PyReadonlyArray2<'py, f64>>() {
                    Ok(f32arr) => f32arr.as_array().mapv(|x| x as f32),
                    Err(_) => {
                        return Err(PyValueError::new_err("Argument 'boxes' needs to be an array of type f32/f64 and shape (n_boxes, 5)!"));
                    }
                };
                boxes_f32_owned.view()
            }
        };
        if detection_boxes.shape()[1] != 5 {
            return Err(PyValueError::new_err(
                "Argument 'boxes' needs to have shape (n_boxes, 5)!",
            ));
        }

        let tracklet_boxes = self.get_tracklet_boxes();

        assign_detections_to_tracks(detection_boxes, tracklet_boxes.view(), self.iou_threshold);

        println!("{}", detection_boxes.sum());
        Ok(Array2::zeros((5, 5)).into_pyarray(_py))
    }
}
