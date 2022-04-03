use std::arch::x86_64::*;
use std::mem::MaybeUninit;
use std::{fmt, ops};

use paste::paste;

use crate::conversion::{VectorConvertInto, VectorTransmuteInto};

trait From256i {
    fn from_256i(x: __m256i) -> Self;
}

trait To256i {
    fn to_256i(self) -> __m256i;
}

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
    ($name: ident, $type: ty, $lanes: expr) => {
        #[derive(Copy, Clone)]
        #[repr(transparent)]
        pub struct $name(pub(crate) __m256i);

        impl VectorTransmuteInto<crate::Float32x8> for $name {
            #[inline(always)]
            fn transmute_vector(self) -> crate::Float32x8 {
                unsafe { crate::Float32x8(_mm256_castsi256_ps(self.0) ) }
            }
        }

        impl VectorTransmuteInto<crate::Float64x4> for $name {
            #[inline(always)]
            fn transmute_vector(self) -> crate::Float64x4 {
                unsafe { crate::Float64x4(_mm256_castsi256_pd(self.0) ) }
            }
        }

        impl VectorTransmuteInto<$name> for crate::Float32x8 {
            #[inline(always)]
            fn transmute_vector(self) -> $name {
                unsafe { $name(_mm256_castps_si256(self.0) ) }
            }
        }

        impl VectorTransmuteInto<$name> for crate::Float64x4 {
            #[inline(always)]
            fn transmute_vector(self) -> $name {
                unsafe { $name(_mm256_castpd_si256(self.0) ) }
            }
        }

        impl From256i for $name {
            #[inline(always)]
            fn from_256i(x: __m256i) -> Self {
                Self(x)
            }
        }

        impl To256i for $name {
            #[inline(always)]
            fn to_256i(self) -> __m256i {
                self.0
            }
        }

        impl $name {
            fn _size_check() {
                unsafe {
                    std::mem::transmute::<[$type; $lanes], [u8; 256 / 8]>([0; $lanes]);
                }
            }

            #[inline(always)]
            #[must_use]
            pub fn zero() -> Self {
                unsafe { Self(_mm256_setzero_si256()) }
            }

            #[inline(always)]
            #[must_use]
            pub fn from_array(array: [$type; $lanes]) -> Self {
                unsafe { Self(_mm256_loadu_si256(array.as_ptr() as *const _)) }
            }

            #[inline(always)]
            #[must_use]
            pub fn to_array(self) -> [$type; $lanes] {
                unsafe {
                    let mut array: MaybeUninit<[$type; $lanes]> = MaybeUninit::uninit();
                    _mm256_storeu_si256(array.as_mut_ptr() as *mut _, self.0);
                    array.assume_init()
                }
            }

            /// Create mask from the most significant bit of each 8-bit element.
            #[inline(always)]
            #[must_use]
            pub fn mask(self) -> u32 {
                unsafe { _mm256_movemask_epi8(self.0) as u32 }
            }

            #[inline(always)]
            #[must_use]
            pub fn andnot(self, rhs: Self) -> Self {
                unsafe { Self(_mm256_andnot_si256(self.0, rhs.0)) }
            }

            #[inline(always)]
            #[must_use]
            pub fn convert<T>(self) -> T where Self: VectorConvertInto<T> {
                <Self as VectorConvertInto<T>>::convert_vector(self)
            }

            #[inline(always)]
            #[must_use]
            pub fn transmute<T>(self) -> T where Self: VectorTransmuteInto<T> {
                <Self as VectorTransmuteInto<T>>::transmute_vector(self)
            }
        }

        impl_operator! { $name, BitAnd, bitand,
            fn bitand(self, rhs: Self) -> Self::Output {
                unsafe { Self(_mm256_and_si256(self.0, rhs.0)) }
            }
        }

        impl_operator! { $name, BitOr, bitor,
            fn bitor(self, rhs: Self) -> Self::Output {
                unsafe { Self(_mm256_or_si256(self.0, rhs.0)) }
            }
        }

        impl_operator! { $name, BitXor, bitxor,
            fn bitxor(self, rhs: Self) -> Self::Output {
                unsafe { Self(_mm256_xor_si256(self.0, rhs.0)) }
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                <[$type; $lanes] as fmt::Debug>::fmt(&self.to_array(), f)
            }
        }
    };
}

make_vector_type!(Int8x32, i8, 32);
make_vector_type!(Uint8x32, u8, 32);

make_vector_type!(Int16x16, i16, 16);
make_vector_type!(Uint16x16, u16, 16);

make_vector_type!(Int32x8, i32, 8);
make_vector_type!(Uint32x8, u32, 8);

make_vector_type!(Int64x4, i64, 4);
make_vector_type!(Uint64x4, u64, 4);

macro_rules! impl_basic_operations {
    (
        $signed: ident, $signed_type: ty, $unsigned: ident, $unsigned_type: ident,
        $splat: ident, $add: ident, $sub: ident, $insert: ident, 
        $cmp_eq: ident, $cmp_gt: ident
    ) => {
        impl_basic_operations!($signed, $signed_type, $splat, $add, $sub, $insert, $cmp_eq);
        impl_basic_operations!($unsigned, $unsigned_type, $splat, $add, $sub, $insert, $cmp_eq);

        impl $signed {
            #[inline(always)]
            #[must_use]
            pub fn gt(self, rhs: Self) -> Self {
                unsafe { Self($cmp_gt(self.0, rhs.0)) }
            }
        }
    };

    (
        $name: ident, $type: ty, $splat: ident, $add: ident,
        $sub: ident, $insert: ident, $cmp_eq: ident
    ) => {
        impl $name {
            #[inline(always)]
            #[must_use]
            pub fn splat(v: $type) -> Self {
                unsafe { Self($splat(v as _)) }
            }

            #[inline(always)]
            #[must_use]
            pub fn eq(self, rhs: Self) -> Self {
                unsafe { Self($cmp_eq(self.0, rhs.0)) }
            }

            #[inline(always)]
            #[must_use]
            pub fn insert<const I: i32>(self, value: $type) -> Self {
                unsafe { Self($insert::<I>(self.0, value as _)) }
            }
        }

        impl_operator! {$name, Add, add,
            fn add(self, rhs: Self) -> Self::Output {
                unsafe { Self($add(self.0, rhs.0)) }
            }
        }

        impl_operator! {$name, Sub, sub,
            fn sub(self, rhs: Self) -> Self::Output {
                unsafe { Self($sub(self.0, rhs.0)) }
            }
        }
    };
}

impl_basic_operations!(
    Int8x32,
    i8,
    Uint8x32,
    u8,
    _mm256_set1_epi8,
    _mm256_add_epi8,
    _mm256_sub_epi8,
    _mm256_insert_epi8,
    _mm256_cmpeq_epi8,
    _mm256_cmpgt_epi8
);

impl_basic_operations!(
    Int16x16,
    i16,
    Uint16x16,
    u16,
    _mm256_set1_epi16,
    _mm256_add_epi16,
    _mm256_sub_epi16,
    _mm256_insert_epi16,
    _mm256_cmpeq_epi16,
    _mm256_cmpgt_epi16
);

impl_basic_operations!(
    Int32x8,
    i32,
    Uint32x8,
    u32,
    _mm256_set1_epi32,
    _mm256_add_epi32,
    _mm256_sub_epi32,
    _mm256_insert_epi32,
    _mm256_cmpeq_epi32,
    _mm256_cmpgt_epi32
);

impl_basic_operations!(
    Int64x4,
    i64,
    Uint64x4,
    u64,
    _mm256_set1_epi64x,
    _mm256_add_epi64,
    _mm256_sub_epi64,
    _mm256_insert_epi64,
    _mm256_cmpeq_epi64,
    _mm256_cmpgt_epi64
);

macro_rules! impl_logical_shifts {
    ($signed: ident, $unsigned: ident, $left_shift: ident, $right_shift: ident) => {
        impl_logical_shifts!($signed, $left_shift, $right_shift);
        impl_logical_shifts!($unsigned, $left_shift, $right_shift);
    };

    ($name: ident, $left_shift: ident, $right_shift: ident) => {
        impl $name {
            #[inline(always)]
            #[must_use]
            pub fn shl<const N: i32>(self) -> Self {
                unsafe { Self($left_shift::<N>(self.0)) }
            }

            #[inline(always)]
            #[must_use]
            pub fn shr_l<const N: i32>(self) -> Self {
                unsafe { Self($right_shift::<N>(self.0)) }
            }
        }
    };
}

impl_logical_shifts!(Int16x16, Uint16x16, _mm256_slli_epi16, _mm256_srli_epi16);
impl_logical_shifts!(Int32x8, Uint32x8, _mm256_slli_epi32, _mm256_srli_epi32);
impl_logical_shifts!(Int64x4, Uint64x4, _mm256_slli_epi64, _mm256_srli_epi64);

macro_rules! impl_arithmetic_shift {
    ($signed: ident, $unsigned: ident, $shift: ident) => {
        impl_arithmetic_shift!($signed, $shift);
        impl_arithmetic_shift!($unsigned, $shift);
    };

    ($name: ident, $shift: ident) => {
        impl $name {
            #[inline(always)]
            #[must_use]
            pub fn shr_a<const N: i32>(self) -> Self {
                unsafe { Self($shift::<N>(self.0)) }
            }
        }
    };
}

impl_arithmetic_shift!(Int16x16, Uint16x16, _mm256_srai_epi16);
impl_arithmetic_shift!(Int32x8, Uint32x8, _mm256_srai_epi32);

macro_rules! impl_comparisons {
    (
        $signed: ident, $unsigned: ident, 
        $signed_max: ident, $signed_min: ident, 
        $unsigned_max: ident, $unsigned_min: ident, 
        $signed_abs: ident
    ) => {
        impl $signed {
            #[inline(always)]
            #[must_use]
            pub fn abs(self) -> Self {
                unsafe { Self($signed_abs(self.0)) }
            }
        }

        impl_comparisons!($signed, $signed_max, $signed_min);
        impl_comparisons!($unsigned, $unsigned_max, $unsigned_min);
    };

    ($name: ident, $max: ident, $min: ident) => {
        impl $name {
            #[inline(always)]
            #[must_use]
            pub fn min(self, rhs: Self) -> Self {
                unsafe { Self($min(self.0, rhs.0)) }
            }

            #[inline(always)]
            #[must_use]
            pub fn max(self, rhs: Self) -> Self {
                unsafe { Self($max(self.0, rhs.0)) }
            }
        }
    };
}

impl_comparisons!(
    Int8x32, 
    Uint8x32, 
    _mm256_max_epi8, 
    _mm256_min_epi8, 
    _mm256_max_epu8, 
    _mm256_min_epu8, 
    _mm256_abs_epi8
);

impl_comparisons!(
    Int16x16, 
    Uint16x16, 
    _mm256_max_epi16, 
    _mm256_min_epi16, 
    _mm256_max_epu16, 
    _mm256_min_epu16, 
    _mm256_abs_epi16
);

impl_comparisons!(
    Int32x8, 
    Uint32x8, 
    _mm256_max_epi32, 
    _mm256_min_epi32, 
    _mm256_max_epu32, 
    _mm256_min_epu32, 
    _mm256_abs_epi32
);

macro_rules! impl_blend {
    ($signed: ident, $unsigned: ident, $blend: ident) => {
        impl_blend!($signed, $blend);
        impl_blend!($unsigned, $blend);
    };

    ($name: ident, $blend: ident) => {
        impl $name {
            #[inline(always)]
            #[must_use]
            pub fn blend<const N: i32>(self, rhs: Self) -> Self {
                unsafe { Self($blend::<N>(self.0, rhs.0)) }
            }
        }
    };
}

impl_blend!(
    Int16x16,
    Uint16x16,
    _mm256_blend_epi16
);

impl_blend!(
    Int32x8,
    Uint32x8,
    _mm256_blend_epi32
);

impl_operator! { Int32x8, Mul, mul,
    fn mul(self, rhs: Self) -> Self {
        unsafe { Self(_mm256_mul_epi32(self.0, rhs.0)) }
    }
}

impl_operator! { Uint32x8, Mul, mul,
    fn mul(self, rhs: Self) -> Self {
        unsafe { Self(_mm256_mul_epu32(self.0, rhs.0)) }
    }
}

macro_rules! impl_signedness_casts {
    ($signed: ident, $unsigned: ident) => {
        impl From<$signed> for $unsigned {
            #[inline(always)]
            fn from(x: $signed) -> Self {
                Self(x.0)
            }
        }

        impl From<$unsigned> for $signed {
            #[inline(always)]
            fn from(x: $unsigned) -> Self {
                Self(x.0)
            }
        }

        impl VectorConvertInto<$signed> for $unsigned {
            #[inline(always)]
            fn convert_vector(self) -> $signed {
                $signed(self.0)
            }
        }

        impl VectorConvertInto<$unsigned> for $signed {
            #[inline(always)]
            fn convert_vector(self) -> $unsigned {
                $unsigned(self.0)
            }
        }
    };
}

impl_signedness_casts!(Int8x32, Uint8x32);
impl_signedness_casts!(Int16x16, Uint16x16);
impl_signedness_casts!(Int32x8, Uint32x8);
impl_signedness_casts!(Int64x4, Uint64x4);

impl VectorConvertInto<crate::Float32x8> for Int32x8 {
    #[inline(always)]
    fn convert_vector(self) -> crate::Float32x8 {
        unsafe { crate::Float32x8(_mm256_cvtepi32_ps(self.0)) }
    }
}

impl<ToV: From256i, FromV: To256i> VectorTransmuteInto<ToV> for FromV {
    #[inline(always)]
    fn transmute_vector(self) -> ToV {
        ToV::from_256i(self.to_256i())
    }
}