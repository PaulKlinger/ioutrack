use ndarray::LinalgScalar;
use num::ToPrimitive;

pub trait BboxNum: PartialOrd + LinalgScalar + ToPrimitive {}
impl<T: PartialOrd + LinalgScalar + ToPrimitive> BboxNum for T {}

#[inline(always)]
pub fn partial_max<T: PartialOrd>(a: T, b: T) -> T {
    if b > a {
        b
    } else {
        a
    }
}

#[inline(always)]
pub fn partial_min<T: PartialOrd>(a: T, b: T) -> T {
    if b > a {
        a
    } else {
        b
    }
}
