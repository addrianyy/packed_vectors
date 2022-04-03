#[cfg(not(target_feature = "avx2"))]
compile_error!("This library requires AVX2 CPU feature.");

mod conversion;

mod float_256;
mod integer_256;

pub use float_256::*;
pub use integer_256::*;
