import numpy as np
import numpy.typing as npt

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
        self, boxes: npt.NDArray[np.float32 | np.float64], return_all: bool
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
