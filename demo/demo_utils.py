import itertools
import os
import random
from collections.abc import Iterable
from typing import Any

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

MOT_COLUMNS = [
    "frame_id",
    "track_id",
    "xmin",
    "ymin",
    "width",
    "height",
    "conf",
    "x",
    "y",
    "z",
]
COLUMN_SUBSET = ["frame_id", "track_id", "xmin", "ymin", "xmax", "ymax"]

COLORS: list[tuple[int, int, int]] = [
    (random.randint(0, 240), random.randint(0, 240), random.randint(0, 240))
    for _ in range(100)
]


def add_max(df: pd.DataFrame) -> None:
    df["xmax"] = df["xmin"] + df["width"]
    df["ymax"] = df["ymin"] + df["height"]


def load_csv(base_path: str, dataset: str, ignore_conf0: bool = False) -> pd.DataFrame:
    csv_path = os.path.join(base_path, dataset, f"{dataset}.txt")
    img_pattern = os.path.join(base_path, "img1", "{:06}.jpg")

    df = pd.read_csv(csv_path, header=None, names=MOT_COLUMNS)
    add_max(df)
    if ignore_conf0:
        df = df[df["conf"] != 0]

    df["xmin"] = df["xmin"].astype(float)
    df["ymin"] = df["ymin"].astype(float)
    add_max(df)
    df = df[COLUMN_SUBSET].copy()
    df["img_path"] = df["frame_id"].map(lambda i: img_pattern.format(i))

    # no (reliable) score, so just add 1.0
    df["score"] = 1.0
    df.set_index("frame_id", inplace=True)
    return df


def read_img(path: str) -> np.ndarray:
    return cv2.imread(path)[..., ::-1]


def plot_bbox(
    img: np.ndarray,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    color: tuple[int, int, int] = (255, 0, 0),
) -> np.ndarray:
    return cv2.rectangle(
        img,
        (int(round(xmin)), int(round(ymin))),
        (int(round(xmax)), int(round(ymax))),
        color,
        2,
    )


def get_annotated_frame(frame_boxes: pd.DataFrame) -> np.ndarray:
    assert frame_boxes["img_path"].nunique() == 1
    img = read_img(frame_boxes["img_path"].iloc[0]).copy()
    for b in frame_boxes.itertuples():
        plot_bbox(
            img,
            b.xmin,
            b.ymin,
            b.xmax,
            b.ymax,
            color=COLORS[int(b.track_id) % len(COLORS)],
        )
    return img


def run_tracker(
    tracker: Any, boxes: pd.DataFrame, return_all: bool = False
) -> pd.DataFrame:
    tracks = []
    frame_paths = boxes["img_path"].to_dict()
    for frame_id in tqdm(range(boxes.index.min(), boxes.index.max() + 1)):
        frame_boxes = boxes.loc[frame_id:frame_id]
        track_boxes = frame_boxes[["xmin", "ymin", "xmax", "ymax", "score"]].values

        frame_tracks = tracker.update(track_boxes, return_all=return_all)
        tracks.append(
            np.concatenate(
                (frame_tracks, frame_id * np.ones((frame_tracks.shape[0], 1))), axis=1
            )
        )

    df = pd.DataFrame(
        np.concatenate(tracks),
        columns=["xmin", "ymin", "xmax", "ymax", "track_id", "frame_id"],
    )
    df["img_path"] = df["frame_id"].map(frame_paths)
    return df.set_index("frame_id")


def write_video(out_path: str, fps: int, frames: Iterable[np.ndarray]) -> None:
    frame_iterator = iter(frames)
    frame = next(frame_iterator)
    h, w, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    for frame in itertools.chain((frame,), frame_iterator):
        writer.write(frame[..., ::-1])

    writer.release()
