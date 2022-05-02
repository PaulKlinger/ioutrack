use num::Float;
use ndarray::prelude::*;
use ndarray::LinalgScalar;
use anyhow::Result;

use crate::ndarray_utils::*;


pub fn intersection_areas<T: LinalgScalar + Float>(boxes1: ArrayView2<T>, boxes2: ArrayView2<T>) -> Result<Array2<T>>{
    let boxes1 = boxes1.insert_axis(Axis(1));
    let boxes2 = boxes2.insert_axis(Axis(0));

    let max_xmin = maximum(boxes1.slice(s![..,..,0]), boxes2.slice(s![..,..,0]))?;
    let max_ymin = maximum(boxes1.slice(s![..,..,1]), boxes2.slice(s![..,..,1]))?;
    let min_xmax = minimum(boxes1.slice(s![..,..,2]), boxes2.slice(s![..,..,2]))?;
    let min_ymax = minimum(boxes1.slice(s![..,..,3]), boxes2.slice(s![..,..,3]))?;

    let intersection_width = (min_xmax - max_xmin).mapv_into(|x| x.max(T::zero()));
    let intersection_height = (min_ymax - max_ymin).mapv_into(|x| x.max(T::zero()));

    Ok(intersection_width * intersection_height)
}

pub fn ious<T: LinalgScalar + Float>(boxes1: ArrayView2<T>, boxes2: ArrayView2<T>) -> Result<Array2<T>>{
    let intersections = intersection_areas(boxes1, boxes2)?;
    let areas1 = (&boxes1.slice(s![..,2]) - &boxes1.slice(s![..,0])) * (&boxes1.slice(s![..,3]) - &boxes1.slice(s![..,1]));
    let areas2 = (&boxes2.slice(s![..,2]) - &boxes2.slice(s![..,0])) * (&boxes2.slice(s![..,3]) - &boxes2.slice(s![..,1]));

    Ok(&intersections / (areas1 + areas2 - &intersections))
}