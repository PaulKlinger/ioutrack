# IOU Track

[![Actions Status](https://github.com/PaulKlinger/ioutrack/workflows/CI/badge.svg)](https://github.com/PaulKlinger/ioutrack/actions)
[![PyPI](https://img.shields.io/pypi/v/ioutrack.svg?style=flat-square)](https://pypi.org/project/ioutrack/)

Python package for IOU-based tracking ([SORT](https://arxiv.org/abs/1602.00763) & [ByteTrack](https://arxiv.org/abs/2110.06864)) written in Rust.


```Python
from ioutrack import Sort

tracker = Sort(max_age=5, min_hits=2)

#                   xmin ymin xmax ymax score
boxes_0 = np.array([[10., 60., 50., 95., 0.8],...])
tracks_0 = tracker.update(boxes_0)

#                            xmin ymin xmax ymax track_id
assert tracks_0 == np.array([[10., 60., 50., 95., 1.],...])
```
Demo video: [https://youtu.be/BLMnY8K9HBE](https://youtu.be/BLMnY8K9HBE)  
Code to generate the demo video is in the [demo](https://github.com/PaulKlinger/ioutrack/tree/main/demo) folder.

Roughly 30x faster than [python/numpy implementation](https://github.com/abewley/sort).
