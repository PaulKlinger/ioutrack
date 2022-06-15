import numpy as np
import numpy.typing as npt
from typing import Union

class BaseTracker:
    def __new__(self) -> BaseTracker: ...
    def update(
        self, boxes: npt.NDArray[Union[np.float32, np.float64]], return_all: bool
    ) -> npt.NDArray[np.float32]: ...
    def get_current_track_boxes(
        self, return_all: bool = False
    ) -> npt.NDArray[np.float32]: ...
    def clear_trackers(self) -> None: ...
    def remove_tracker(self, track_id: int) -> None: ...

class Sort(BaseTracker):
    max_age: int
    min_hits: int
    iou_threshold: float
    init_tracker_min_score: float
    tracklets: dict[int, KalmanBoxTracker]
    n_steps: int

    def __new__(
        self,
        max_age: int = 1,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        init_tracker_min_score: float = 0.0,
        measurement_noise: tuple[float, float, float, float] = (1.0, 1.0, 10.0, 0.05),
        process_noise: tuple[float, float, float, float, float, float, float] = (
            1.0,
            1.0,
            1.0,
            0.001,
            0.01,
            0.01,
            0.0001,
        ),
    ) -> Sort:
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
        measurement_noise
            Diagonal of the measurement noise covariance matrix
            i.e. uncertainties of (x, y, s, r) measurements
            defaults should be reasonable in most cases
         process_noise
            Diagonal of the process noise covariance matrix
            i.e. uncertainties of (x, y, s, r, dx, dy, ds) during each step
            defaults should be reasonable in most cases
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
    def get_current_track_boxes(
        self, return_all: bool = False
    ) -> npt.NDArray[np.float32]:
        """Return current track boxes

        Parameters
        ----------
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
    def clear_trackers(self) -> None:
        """Remove all tracklets"""
        ...
    def remove_tracker(self, track_id: int) -> None:
        """Remove tracklet with the given track_id,
        do nothing if it doesn't exist"""
        ...

class ByteTrack(BaseTracker):
    max_age: int
    min_hits: int
    iou_threshold: float
    init_tracker_min_score: float
    tracklets: dict[int, KalmanBoxTracker]
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
        process_noise: tuple[float, float, float, float, float, float, float] = (
            1.0,
            1.0,
            1.0,
            0.001,
            0.01,
            0.01,
            0.0001,
        ),
    ) -> ByteTrack:
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
        measurement_noise
            Diagonal of the measurement noise covariance matrix
            i.e. uncertainties of (x, y, s, r) measurements
            defaults should be reasonable in most cases
        process_noise
            Diagonal of the process noise covariance matrix
            i.e. uncertainties of (x, y, s, r, dx, dy, ds) during each step
            defaults should be reasonable in most cases
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
    def get_current_track_boxes(
        self, return_all: bool = False
    ) -> npt.NDArray[np.float32]:
        """Return current track boxes

        Parameters
        ----------
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
    def clear_trackers(self) -> None:
        """Remove all tracklets"""
        ...
    def remove_tracker(self, track_id: int) -> None:
        """Remove tracklet with the given track_id,
        do nothing if it doesn't exist"""
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
