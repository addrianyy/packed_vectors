use std::arch::x86_64::*;
use std::mem::MaybeUninit;
use std::{fmt, ops};

use paste::paste;

use crate::conversion::{VectorConvertInto, VectorTransmuteInto};

macro_rules! impl_operator {
    ($name: ident, $op: ident, $op_function: ident, $function: item) => {
        impl ops::$op for $name {
            type Output = Self;

            #[inline(always)]
            #[must_use]
            $function
        }

        paste! {
            impl ops::[<$op Assign>] for $name {
                #[inline(always)]
                fn [<$op_function _assign>](&mut self, rhs: Self) {
                    *self = <Self as ops::$op>::$op_function(*self, rhs);
                }
            }
        }
    }
}

macro_rules! make_vector_type {
    ($name: ident, $type: ty, $lanes: expr, $avx_type: ty, $postfix: ident) => {
        #[derive(Copy, Clone)]
        #[repr(transparent)]
        pub struct $name(pub(crate) $avx_type);

        macro_rules! intrinsic {
            ($function: ident) => {
                paste! { [< $function _ $postfix>] }
            };
        }

        macro_rules! comparison {
            ($comparison_name: ident, $comparison_constant: ident) => {
                #[inline(always)]
                #[must_use]
                pub fn $comparison_name(self, rhs: Self) -> Self {
                    unsafe {
                        paste! {
                            Self([<_mm256_cmp _ $postfix>]::<$comparison_constant>(self.0, rhs.0))
                        }
                    }
                }
            };
        }

        impl $name {
            fn _size_check() {
                unsafe {
                    std::mem::transmute::<[$type; $lanes], [u8; 256 / 8]>([0.0; $lanes]);
                }
            }

            comparison!(eq, _CMP_EQ_OQ);
            comparison!(ne, _CMP_NEQ_OQ);

            comparison!(gt, _CMP_GT_OQ);
            comparison!(lt, _CMP_LT_OQ);

            comparison!(ge, _CMP_GE_OQ);
            comparison!(le, _CMP_LE_OQ);

            #[inline(always)]
            #[must_use]
            pub fn zero() -> Self {
                unsafe { Self(intrinsic!(_mm256_setzero)()) }
            }

            #[inline(always)]
            #[must_use]
            pub fn splat(v: $type) -> Self {
                unsafe { Self(intrinsic!(_mm256_set1)(v)) }
            }

            #[inline(always)]
            #[must_use]
            pub fn from_array(array: [$type; $lanes]) -> Self {
                unsafe { Self(intrinsic!(_mm256_loadu)(array.as_ptr() as *const _)) }
            }

            #[inline(always)]
            #[must_use]
            pub fn to_array(self) -> [$type; $lanes] {
                unsafe {
                    let mut array: MaybeUninit<[$type; $lanes]> = MaybeUninit::uninit();
                    intrinsic!(_mm256_storeu)(array.as_mut_ptr() as *mut _, self.0);
                    array.assume_init()
                }
            }

            /// Set each bit of mask based on the most significant bit of the corresponding packed
            /// floating-point element.
            #[inline(always)]
            #[must_use]
            pub fn mask(self) -> u32 {
                unsafe { intrinsic!(_mm256_movemask)(self.0) as u32 }
            }

            /// ~self & rhs
            #[inline(always)]
            #[must_use]
            pub fn andnot(self, rhs: Self) -> Self {
                unsafe { Self(intrinsic!(_mm256_andnot)(self.0, rhs.0)) }
            }

            #[inline(always)]
            #[must_use]
            pub fn min(self, rhs: Self) -> Self {
                unsafe { Self(intrinsic!(_mm256_min)(self.0, rhs.0)) }
            }

            #[inline(always)]
            #[must_use]
            pub fn max(self, rhs: Self) -> Self {
                unsafe { Self(intrinsic!(_mm256_max)(self.0, rhs.0)) }
            }

            #[inline(always)]
            #[must_use]
            pub fn blend<const I: i32>(self, rhs: Self) -> Self {
                unsafe {
                    paste! {
                        Self([<_mm256_blend _ $postfix>]::<I>(self.0, rhs.0))
                    }
                }
            }

            #[inline(always)]
            #[must_use]
            pub fn floor(self) -> Self {
                unsafe { Self(intrinsic!(_mm256_floor)(self.0)) }
            }

            #[inline(always)]
            #[must_use]
            pub fn ceil(self) -> Self {
                unsafe { Self(intrinsic!(_mm256_ceil)(self.0)) }
            }

            #[inline(always)]
            #[must_use]
            pub fn trunc(self) -> Self {
                // _MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC
                unsafe {
                    paste! {
                        Self([<_mm256_round _ $postfix>]::<0x0b>(self.0))
                    }
                }
            }

            #[inline(always)]
            #[must_use]
            pub fn round(self) -> Self {
                // _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC
                unsafe {
                    paste! {
                        Self([<_mm256_round _ $postfix>]::<0x08>(self.0))
                    }
                }
            }

            #[inline(always)]
            #[must_use]
            pub fn sqrt(self) -> Self {
                unsafe { Self(intrinsic!(_mm256_sqrt)(self.0)) }
            }

            /// (self * b) + c
            #[cfg(target_feature = "fma")]
            #[inline(always)]
            #[must_use]
            pub fn fmadd(self, b: Self, c: Self) -> Self {
                unsafe { Self(intrinsic!(_mm256_fmadd)(self.0, b.0, c.0)) }
            }

            /// (self * b) - c
            #[cfg(target_feature = "fma")]
            #[inline(always)]
            #[must_use]
            pub fn fmsub(self, b: Self, c: Self) -> Self {
                unsafe { Self(intrinsic!(_mm256_fmsub)(self.0, b.0, c.0)) }
            }

            #[inline(always)]
            #[must_use]
            pub fn convert<T>(self) -> T
            where
                Self: VectorConvertInto<T>,
            {
                <Self as VectorConvertInto<T>>::convert_vector(self)
            }

            #[inline(always)]
            #[must_use]
            pub fn transmute<T>(self) -> T
            where
                Self: VectorTransmuteInto<T>,
            {
                <Self as VectorTransmuteInto<T>>::transmute_vector(self)
            }
        }

        impl_operator! { $name, Add, add,
            fn add(self, rhs: Self) -> Self::Output {
                unsafe { Self(intrinsic!(_mm256_add)(self.0, rhs.0)) }
            }
        }

        impl_operator! { $name, Sub, sub,
            fn sub(self, rhs: Self) -> Self::Output {
                unsafe { Self(intrinsic!(_mm256_sub)(self.0, rhs.0)) }
            }
        }

        impl_operator! { $name, Mul, mul,
            fn mul(self, rhs: Self) -> Self::Output {
                unsafe { Self(intrinsic!(_mm256_mul)(self.0, rhs.0)) }
            }
        }

        impl_operator! { $name, Div, div,
            fn div(self, rhs: Self) -> Self::Output {
                unsafe { Self(intrinsic!(_mm256_div)(self.0, rhs.0)) }
            }
        }

        impl_operator! { $name, BitAnd, bitand,
            fn bitand(self, rhs: Self) -> Self::Output {
                unsafe { Self(intrinsic!(_mm256_and)(self.0, rhs.0)) }
            }
        }

        impl_operator! { $name, BitOr, bitor,
            fn bitor(self, rhs: Self) -> Self::Output {
                unsafe { Self(intrinsic!(_mm256_or)(self.0, rhs.0)) }
            }
        }

        impl_operator! { $name, BitXor, bitxor,
            fn bitxor(self, rhs: Self) -> Self::Output {
                unsafe { Self(intrinsic!(_mm256_xor)(self.0, rhs.0)) }
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                <[$type; $lanes] as fmt::Debug>::fmt(&self.to_array(), f)
            }
        }
    };
}

make_vector_type!(Float32x8, f32, 8, __m256, ps);
make_vector_type!(Float64x4, f64, 4, __m256d, pd);

impl Float32x8 {
    pub fn rsqrt(self) -> Self {
        unsafe { Self(_mm256_rsqrt_ps(self.0)) }
    }
}

impl VectorConvertInto<crate::Int32x8> for Float32x8 {
    #[inline(always)]
    #[must_use]
    fn convert_vector(self) -> crate::Int32x8 {
        unsafe { crate::Int32x8(_mm256_cvtps_epi32(self.0)) }
    }
}
