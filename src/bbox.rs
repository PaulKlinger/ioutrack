use ndarray::prelude::*;
use ndarray::Zip;
use crate::num_utils::{BboxNum, partial_max};

use crate::ndarray_utils::*;


/// Calculate areas of intersections of all combinations of a box from boxes1 and one from boxes2
/// Both are in format [[xmin1, ymin1, xmax1, ymax1], [xmin2,...],...]
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
pub fn intersection_areas<T: BboxNum>(boxes1: ArrayView2<T>, boxes2: ArrayView2<T>) -> Array2<T>{
    let boxes1 = boxes1.insert_axis(Axis(1));
    let boxes2 = boxes2.insert_axis(Axis(0));

    // we know that these are broadcastable, so unwrap
    // TODO: we know the output shape here, so we don't need to compute it every time...
    let max_xmin = maximum(boxes1.slice(s![..,..,0]), boxes2.slice(s![..,..,0])).unwrap();
    let max_ymin = maximum(boxes1.slice(s![..,..,1]), boxes2.slice(s![..,..,1])).unwrap();
    let min_xmax = minimum(boxes1.slice(s![..,..,2]), boxes2.slice(s![..,..,2])).unwrap();
    let min_ymax = minimum(boxes1.slice(s![..,..,3]), boxes2.slice(s![..,..,3])).unwrap();

    let intersection_width = (min_xmax - max_xmin).mapv_into(|x| partial_max(x, T::zero()));
    let intersection_height = (min_ymax - max_ymin).mapv_into(|x| partial_max(x, T::zero()));

    // We know this is 2 dim (len(boxes_1), len(boxes_2))
    // broadcasting is implemented manually in maximum/minimum, so this is not automatic
    (intersection_width * intersection_height).into_dimensionality().unwrap()
}


/// Calculate IntersectionOverUnion of all combinations of a box from boxes1 and one from boxes2
/// Both are in format [[xmin1, ymin1, xmax1, ymax1], [xmin2,...],...]
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
pub fn ious<T: BboxNum>(boxes1: ArrayView2<T>, boxes2: ArrayView2<T>) -> Array2<f64>{
    let intersections = intersection_areas(boxes1, boxes2);
    let areas1 = (&boxes1.slice(s![..,2]) - &boxes1.slice(s![..,0])) * (&boxes1.slice(s![..,3]) - &boxes1.slice(s![..,1]));
    let areas2 = (&boxes2.slice(s![..,2]) - &boxes2.slice(s![..,0])) * (&boxes2.slice(s![..,3]) - &boxes2.slice(s![..,1]));

    let mut out: Array2<f64> = Array2::zeros(intersections.raw_dim());
    Zip::from(&mut out)
        .and(&intersections)
        .and_broadcast(&areas1.insert_axis(Axis(1)))
        .and_broadcast(&areas2.insert_axis(Axis(0)))
        .for_each(|o, &intersection, &area1, &area2|
            {*o = intersection.to_f64().unwrap() / (area1 + area2 - intersection).to_f64().unwrap()});
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_intersection_areas() {
        let intersections = intersection_areas(
             (array![[0., 0., 10., 12.], [9., 11., 13., 14.]]).view(),
             (array![[5., 7., 13., 11.]]).view()
         );
         assert_eq!(intersections, array![[20.], [0.]]);
    }

    #[test]
    fn test_intersection_areas_int() {
        let intersections = intersection_areas(
             (array![[0, 0, 10, 12], [9, 11, 13, 14]]).view(),
             (array![[5, 7, 13, 11]]).view()
         );
         assert_eq!(intersections, array![[20], [0]]);
    }

    #[test]
    fn test_ious() {
        let iou_result = ious(
            (array![[0., 0., 10., 12.]]).view(),
            (array![[5., 7., 13., 11.], [9., 11., 13., 14.]]).view()
        );
        assert_abs_diff_eq!(
            iou_result,
            array![[0.15151, 0.00763]],
            epsilon = 0.0001
        );

        let iou_result = ious(
            (array![[0., 0., 10., 12.], [5., 7., 13., 11.]]).view(),
            (array![[5., 7., 13., 11.]]).view()
        );
        assert_abs_diff_eq!(
            iou_result,
            array![[0.15151], [1.]],
            epsilon = 0.0001
        );
    }

    #[test]
    fn test_ious_int() {
        let iou_result = ious(
            (array![[0, 0, 10, 12]]).view(),
            (array![[5, 7, 13, 11], [9, 11, 13, 14]]).view()
        );
        assert_abs_diff_eq!(
            iou_result,
            array![[0.15151, 0.00763]],
            epsilon = 0.0001
        );
    }
}