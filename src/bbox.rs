use crate::num_utils::{partial_max, BboxNum};
use approx::AbsDiffEq;
use ndarray::prelude::*;
use ndarray::Zip;
use num::Float;

use crate::ndarray_utils::*;

#[derive(Debug, Clone, PartialEq)]
pub struct Bbox<T: BboxNum> {
    pub xmin: T,
    pub ymin: T,
    pub xmax: T,
    pub ymax: T,
}

impl<T: BboxNum> Bbox<T> {
    /// Convert Bbox to center_x, center_y, area, aspect_ratio representation
    /// (measurement for Kalman filter)
    pub fn to_z(&self) -> Array1<f32> {
        let width = (self.xmax - self.xmin).to_f32().unwrap();
        let height = (self.ymax - self.ymin).to_f32().unwrap();
        array![
            (self.xmax + self.xmin).to_f32().unwrap() / 2.,
            (self.ymax + self.ymin).to_f32().unwrap() / 2.,
            width * height,
            width / height
        ]
    }

    pub fn to_bounds(&self) -> [T; 4] {
        [self.xmin, self.ymin, self.xmax, self.ymax]
    }
}

impl<'a, T: BboxNum> TryFrom<ArrayView1<'a, T>> for Bbox<T> {
    type Error = &'static str;
    fn try_from(bounds: ArrayView1<'a, T>) -> Result<Bbox<T>, Self::Error> {
        match bounds.dim() {
            4 => Ok(Bbox {
                xmin: bounds[0],
                ymin: bounds[1],
                xmax: bounds[2],
                ymax: bounds[3],
            }),
            _ => Err("Array must have 4 elements to convert to bbox!"),
        }
    }
}

impl<T> Bbox<T>
where
    T: BboxNum + Float,
    f32: Into<T>,
{
    /// Convert center_x, center_y, area, aspect_ratio representation to Bbox
    pub fn from_z(z: ArrayView1<T>) -> anyhow::Result<Self> {
        if z.dim() != 4 {
            return Err(anyhow::anyhow!("z vector must have exactly 4 elements!"));
        }
        // area = width * height = width**2 / aspect
        let width = (z[2] * z[3]).sqrt();
        let height = width / z[3];
        Ok(Self {
            xmin: z[0] - width / 2.0.into(),
            xmax: z[0] + width / 2.0.into(),
            ymin: z[1] - height / 2.0.into(),
            ymax: z[1] + height / 2.0.into(),
        })
    }
}

impl<T: AbsDiffEq + BboxNum> AbsDiffEq for Bbox<T>
where
    T::Epsilon: Copy,
{
    type Epsilon = T::Epsilon;

    fn default_epsilon() -> T::Epsilon {
        T::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: T::Epsilon) -> bool {
        T::abs_diff_eq(&self.xmin, &other.xmin, epsilon)
            && T::abs_diff_eq(&self.ymin, &other.ymin, epsilon)
            && T::abs_diff_eq(&self.xmax, &other.xmax, epsilon)
            && T::abs_diff_eq(&self.ymax, &other.ymax, epsilon)
    }
}

/// Calculate areas of intersections of all combinations of a box from boxes1 and one from boxes2
/// Both are in format [[xmin1, ymin1, xmax1, ymax1], [xmin2,...],...]
///
/// boxes1 shape = (n, 4)
/// boxes2 shape = (m, 4)
/// result shape = (n, m)
///
/// # Examples
///
/// ```
/// use ndarray::prelude::array;
/// let intersections = ioutrack::bbox::intersection_areas(
///     (array![[0., 0., 10., 12.]]).view(),
///     (array![[5., 7., 13., 11.], [9., 11., 13., 14.]]).view()
/// );
/// assert_eq!(intersections, array![[20., 1.]]);
/// ```
pub fn intersection_areas<T: BboxNum>(boxes1: ArrayView2<T>, boxes2: ArrayView2<T>) -> Array2<T> {
    let boxes1 = boxes1.insert_axis(Axis(1));
    let boxes2 = boxes2.insert_axis(Axis(0));

    // we know that these are broadcastable, so unwrap
    // TODO: we know the output shape here, so we don't need to compute it every time...
    let max_xmin = maximum(boxes1.slice(s![.., .., 0]), boxes2.slice(s![.., .., 0])).unwrap();
    let max_ymin = maximum(boxes1.slice(s![.., .., 1]), boxes2.slice(s![.., .., 1])).unwrap();
    let min_xmax = minimum(boxes1.slice(s![.., .., 2]), boxes2.slice(s![.., .., 2])).unwrap();
    let min_ymax = minimum(boxes1.slice(s![.., .., 3]), boxes2.slice(s![.., .., 3])).unwrap();

    let intersection_width = (min_xmax - max_xmin).mapv_into(|x| partial_max(x, T::zero()));
    let intersection_height = (min_ymax - max_ymin).mapv_into(|x| partial_max(x, T::zero()));

    // We know this is 2 dim (len(boxes_1), len(boxes_2))
    // broadcasting is implemented manually in maximum/minimum, so this is not automatic
    (intersection_width * intersection_height)
        .into_dimensionality()
        .unwrap()
}

/// Calculate IntersectionOverUnion of all combinations of a box from boxes1 and one from boxes2
/// Both are in format [[xmin1, ymin1, xmax1, ymax1, (score)], [xmin2,...],...]
/// boxes1 shape = (n, 4/5)
/// boxes2 shape = (m, 4/5)
/// result shape = (n, m)
///
/// # Examples
///
/// ```
/// use ndarray::prelude::array;
/// use approx::assert_abs_diff_eq;
/// let iou_result = ioutrack::bbox::ious(
///     (array![[0., 0., 10., 12.], [5., 7., 13., 11.]]).view(),
///     (array![[5., 7., 13., 11.], [9., 11., 13., 14.]]).view()
/// );
/// assert_abs_diff_eq!(
///     iou_result,
///     array![[0.15151, 0.00763],
///            [1.,      0.]],
///     epsilon = 0.0001
/// );
/// ```
pub fn ious<T: BboxNum>(boxes1: ArrayView2<T>, boxes2: ArrayView2<T>) -> Array2<f32> {
    let intersections = intersection_areas(boxes1, boxes2);
    let areas1 = (&boxes1.slice(s![.., 2]) - &boxes1.slice(s![.., 0]))
        * (&boxes1.slice(s![.., 3]) - &boxes1.slice(s![.., 1]));
    let areas2 = (&boxes2.slice(s![.., 2]) - &boxes2.slice(s![.., 0]))
        * (&boxes2.slice(s![.., 3]) - &boxes2.slice(s![.., 1]));

    let mut out: Array2<f32> = Array2::zeros(intersections.raw_dim());
    Zip::from(&mut out)
        .and(&intersections)
        .and_broadcast(&areas1.insert_axis(Axis(1)))
        .and_broadcast(&areas2.insert_axis(Axis(0)))
        .for_each(|o, &intersection, &area1, &area2| {
            *o = intersection.to_f32().unwrap() / (area1 + area2 - intersection).to_f32().unwrap()
        });
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_bbox_to_z() {
        let bbox = Bbox::<u32> {
            xmin: 0,
            ymin: 10,
            xmax: 20,
            ymax: 20,
        };

        assert_abs_diff_eq!(bbox.to_z(), array![10., 15., 200., 2.], epsilon = 0.0001);
    }

    #[test]
    fn test_z_to_bbox() {
        let bbox = Bbox::from_z(array![10., 15., 200., 2.].view()).unwrap();

        assert_abs_diff_eq!(
            bbox,
            Bbox::<f32> {
                xmin: 0.,
                ymin: 10.,
                xmax: 20.,
                ymax: 20.,
            },
            epsilon = 0.0001
        );
    }

    #[test]
    fn test_intersection_areas() {
        let intersections = intersection_areas(
            (array![[0., 0., 10., 12.], [9., 11., 13., 14.]]).view(),
            (array![[5., 7., 13., 11.]]).view(),
        );
        assert_eq!(intersections, array![[20.], [0.]]);
    }

    #[test]
    fn test_intersection_areas_int() {
        let intersections = intersection_areas(
            (array![[0, 0, 10, 12], [9, 11, 13, 14]]).view(),
            (array![[5, 7, 13, 11]]).view(),
        );
        assert_eq!(intersections, array![[20], [0]]);
    }

    #[test]
    fn test_ious() {
        let iou_result = ious(
            (array![[0., 0., 10., 12.]]).view(),
            (array![[5., 7., 13., 11.], [9., 11., 13., 14.]]).view(),
        );
        assert_abs_diff_eq!(iou_result, array![[0.15151, 0.00763]], epsilon = 0.0001);

        let iou_result = ious(
            (array![[0., 0., 10., 12.], [5., 7., 13., 11.]]).view(),
            (array![[5., 7., 13., 11.]]).view(),
        );
        assert_abs_diff_eq!(iou_result, array![[0.15151], [1.]], epsilon = 0.0001);
    }

    #[test]
    fn test_ious_int() {
        let iou_result = ious(
            (array![[0, 0, 10, 12]]).view(),
            (array![[5, 7, 13, 11], [9, 11, 13, 14]]).view(),
        );
        assert_abs_diff_eq!(iou_result, array![[0.15151, 0.00763]], epsilon = 0.0001);
    }
}
