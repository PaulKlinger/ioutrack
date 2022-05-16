use num::cast;
use numpy::ndarray::prelude::*;
use numpy::pyo3::exceptions::PyValueError;
use numpy::pyo3::prelude::*;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use std::collections::HashMap;

use lapjv::lapjv_rect;

use crate::bbox::{ious, Bbox};
use crate::box_tracker::{KalmanBoxTracker, KalmanBoxTrackerParams};

type TrackidBoxes = Vec<(u32, Bbox<f32>)>;
type ScoreBoxes = Vec<(f32, Bbox<f32>)>;

/// Assign detection boxes to track boxes
///
/// Parameters
/// ----------
/// detections
///     detection boxes, shape (n_detections, 5)
///     of the form [[xmin1, ymin1, xmax1, ymax1, score1], [xmin2,...],...]
/// tracks
///     track boxes, shape (n_tracks, 5)
///     of the form [[xmin1, ymin1, xmax1, ymax1, track_id], [xmin2,...],...]
///
/// Returns
/// -------
/// Tuple of (matches, unmatched_detections)
/// where matches = [(track_id, bbox),...]
/// and unmatched_detections = [(score, bbox),...]
fn assign_detections_to_tracks(
    detections: ArrayView2<f32>,
    tracks: ArrayView2<f32>,
    iou_threshold: f32,
) -> anyhow::Result<(TrackidBoxes, ScoreBoxes)> {
    let mut det_track_ious = ious(detections, tracks);
    det_track_ious.mapv_inplace(|x| -x);
    let (track_idxs, _) = lapjv_rect(det_track_ious.view())?;

    let mut match_updates = Vec::new();
    let mut unmatched_dets = Vec::new();
    for (det_idx, &maybe_track_idx) in track_idxs.iter().enumerate() {
        let det_box = detections.slice(s![det_idx, 0..4]).try_into().unwrap();
        let score = detections[(det_idx, 4)];
        match maybe_track_idx {
            Some(track_idx) => {
                // we negated the ious, so negate again here
                if -det_track_ious[(det_idx, track_idx)] > iou_threshold {
                    match_updates.push((tracks[(track_idx, 4)] as u32, det_box))
                } else {
                    unmatched_dets.push((score, det_box));
                }
            }
            None => unmatched_dets.push((score, det_box)),
        }
    }

    Ok((match_updates, unmatched_dets))
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
/// init_score_threshold
///     minimum score to create a new tracklet from unmatched detection box
#[pyclass(text_signature = "(max_age=1, min_hits=3, iou_threshold=0.3, init_score_threshold=0.0)")]
pub struct SORTTracker {
    #[pyo3(get, set)]
    pub max_age: u32,
    #[pyo3(get, set)]
    pub min_hits: u32,
    #[pyo3(get, set)]
    pub iou_threshold: f32,
    #[pyo3(get, set)]
    pub init_score_threshold: f32,
    /// id of next tracklet initialized
    next_track_id: u32,
    /// current tracklets
    #[pyo3(get)]
    pub tracklets: HashMap<u32, KalmanBoxTracker>,
    /// number of steps the tracker has run for
    #[pyo3(get)]
    pub n_steps: u32,
}

impl SORTTracker {
    fn predict(&mut self) -> Array2<f32> {
        let mut data = Vec::with_capacity(self.tracklets.len() * 5);
        for (_, tracklet) in self.tracklets.iter_mut() {
            let b = tracklet.predict();
            data.extend(b.to_bounds());
            data.push(cast(tracklet.id).unwrap());
        }
        Array2::from_shape_vec((self.tracklets.len(), 5), data).unwrap()
    }

    fn get_tracklet_boxes(&self, return_all: bool) -> Array2<f32> {
        let mut data = Vec::new();
        for (_, tracklet) in self.tracklets.iter() {
            if return_all || (tracklet.hit_streak >= self.min_hits || self.n_steps < self.min_hits)
            {
                data.extend(tracklet.bbox().to_bounds());
                data.push(cast(tracklet.id).unwrap());
            }
        }
        Array2::from_shape_vec((data.len() / 5, 5), data).unwrap()
    }

    fn create_tracklets(&mut self, score_boxes: ScoreBoxes) {
        for (score, bbox) in score_boxes {
            if score >= self.init_score_threshold {
                self.tracklets.insert(
                    self.next_track_id,
                    KalmanBoxTracker::new(KalmanBoxTrackerParams {
                        id: self.next_track_id,
                        bbox,
                        center_var: None,
                        area_var: None,
                        aspect_var: None,
                    }),
                );
                self.next_track_id += 1
            }
        }
    }

    fn update(
        &mut self,
        detection_boxes: CowArray<f32, Ix2>,
        return_all: bool,
    ) -> anyhow::Result<Array2<f32>> {
        let tracklet_boxes = self.predict();

        let (matched_boxes, unmatched_detections) = assign_detections_to_tracks(
            detection_boxes.view(),
            tracklet_boxes.view(),
            self.iou_threshold,
        )?;

        for (track_id, bbox) in matched_boxes {
            self.tracklets.get_mut(&track_id).unwrap().update(bbox)?;
        }

        // remove stale tracklets
        self.tracklets
            .retain(|_, tracklet| tracklet.steps_since_update <= self.max_age);

        self.create_tracklets(unmatched_detections);

        self.n_steps += 1;
        Ok(self.get_tracklet_boxes(return_all))
    }
}

#[pymethods]
impl SORTTracker {
    #[new]
    #[args(
        max_age = "1",
        min_hits = "3",
        iou_threshold = "0.3",
        init_score_threshold = "0.0"
    )]
    fn py_new(max_age: u32, min_hits: u32, iou_threshold: f32, init_score_threshold: f32) -> Self {
        SORTTracker {
            max_age,
            min_hits,
            iou_threshold,
            init_score_threshold,
            next_track_id: 1,
            tracklets: HashMap::new(),
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
    #[pyo3(name = "update", text_signature = "(boxes, return_all = False)")]
    fn py_update<'py>(
        &mut self,
        _py: Python<'py>,
        boxes: &'py PyAny,
        return_all: bool,
    ) -> PyResult<&'py PyArray2<f32>> {
        // We allow 'boxes' to be either f32 (then we use it directly) or f64 (then we convert to f32)
        // TODO: find some way to extract this into a function...
        let boxes_py32_res: PyResult<PyReadonlyArray2<'py, f32>> = boxes.extract();
        let detection_boxes: CowArray<f32, Ix2> = match boxes_py32_res {
            Ok(ref arr) => arr.as_array().into(),
            Err(_) => boxes
                .extract::<PyReadonlyArray2<'py, f64>>()
                .map_err(|_| PyValueError::new_err("Argument 'boxes' needs to be an array of type f32/f64 and shape (n_boxes, 5)!",))?
                .as_array()
                .mapv(|x| x as f32)
                .into(),
        };
        if detection_boxes.shape()[1] != 5 {
            return Err(PyValueError::new_err(
                "Argument 'boxes' needs to have shape (n_boxes, 5)!",
            ));
        }

        return Ok(self.update(detection_boxes, return_all)?.into_pyarray(_py));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_first_update() {
        let mut tracker = SORTTracker::py_new(1, 3, 0.3, 0.3);
        assert_abs_diff_eq!(
            tracker
                .update(
                    array![[0.0, 1.5, 12.6, 25.0, 0.9], [-5.5, 18.0, 1.0, 20.0, 0.15]].into(),
                    false
                )
                .unwrap(),
            array![[0.0, 1.5, 12.6, 25.0, 1.0]],
            epsilon = 0.00001
        )
    }
}
