import numpy as np
import numpy.typing as npt
from typing import Union

class SORTTracker:
    max_age: int
    min_hits: int
    iou_threshold: float
    init_tracker_min_score: float
    tracklets: list[KalmanBoxTracker]
    n_steps: int

    def __new__(
        self,
        max_age: int = 1,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        init_tracker_min_score: float = 0.0,
        measurement_noise: tuple[float, float, float, float] = (1.0, 1.0, 10.0, 0.05),
        process_noiselist: tuple[float, float, float, float, float, float, float] = (
            1.0,
            1.0,
            1.0,
            0.001,
            0.01,
            0.01,
            0.0001,
        ),
    ) -> SORTTracker:
        """Create a new SORT bbox tracker

        Parameters
        ----------
        max_age
            maximum frames a tracklet is kept alive without matching detections
        min_hits
            minimum number of successive detections before a tracklet is set to alive
        iou_threshold
            minimum IOU to assign detection to tracklet
        init_tracker_min_score
            minimum score to create a new tracklet from unmatched detection box
        """
        ...
    def update(
        self, boxes: npt.NDArray[Union[np.float32, np.float64]], return_all: bool
    ) -> npt.NDArray[np.float32]:
        """Update the tracker with new boxes and return position of current tracklets

        Parameters
        ----------
        boxes
            array of boxes of shape (n_boxes, 5)
            of the form [[xmin1, ymin1, xmax1, ymax1, score1], [xmin2,...],...]
        return_all
            if true return all living trackers, including inactive (but not dead) ones
            otherwise return only active trackers (those that got at least min_hits
            matching boxes in a row)

        Returns
        -------
            array of tracklet boxes with shape (n_tracks, 5)
            of the form [[xmin1, ymin1, xmax1, ymax1, track_id1], [xmin2,...],...]
        """
        ...

class ByteTrack:
    max_age: int
    min_hits: int
    iou_threshold: float
    init_tracker_min_score: float
    tracklets: list[KalmanBoxTracker]
    n_steps: int

    def __new__(
        self,
        max_age: int = 1,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        init_tracker_min_score: float = 0.8,
        high_score_threshold: float = 0.7,
        low_score_threshold: float = 0.1,
        measurement_noise: tuple[float, float, float, float] = (1.0, 1.0, 10.0, 0.05),
        process_noiselist: tuple[float, float, float, float, float, float, float] = (
            1.0,
            1.0,
            1.0,
            0.001,
            0.01,
            0.01,
            0.0001,
        ),
    ) -> SORTTracker:
        """Create a new SORT bbox tracker

        Parameters
        ----------
        max_age
            maximum frames a tracklet is kept alive without matching detections
        min_hits
            minimum number of successive detections before a tracklet is set to alive
        iou_threshold
            minimum IOU to assign detection to tracklet
        init_tracker_min_score
            minimum score to create a new tracklet from unmatched detection box
        high_score_threshold
            boxes with higher scores than this will be used in the first round of association
        low_score_threshold
            boxes with score between low_score_threshold and high_score_threshold
            will be used in the second round of association
        measurement_variance
            diagonal of the measurement noise covariance matrix
            i.e. measurement uncertainty of bbox (x, y, area, aspect_ratio)
        process_variance
            diagonal of the process noise covariance matrix
            i.e. uncertainty in the step transition of (x, y, area, aspect_ratio, x_vel, y_vel, area_vel)
        """
        ...
    def update(
        self, boxes: npt.NDArray[Union[np.float32, np.float64]], return_all: bool
    ) -> npt.NDArray[np.float32]:
        """Update the tracker with new boxes and return position of current tracklets

        Parameters
        ----------
        boxes
            array of boxes of shape (n_boxes, 5)
            of the form [[xmin1, ymin1, xmax1, ymax1, score1], [xmin2,...],...]
        return_all
            if true return all living trackers, including inactive (but not dead) ones
            otherwise return only active trackers (those that got at least min_hits
            matching boxes in a row)

        Returns
        -------
            array of tracklet boxes with shape (n_tracks, 5)
            of the form [[xmin1, ymin1, xmax1, ymax1, track_id1], [xmin2,...],...]
        """
        ...

class Bbox:
    xmin: float
    ymin: float
    xmax: float
    ymax: float

class KalmanBoxTracker:
    id: int
    age: int
    hits: int
    hit_streak: int
    steps_since_hit: int
