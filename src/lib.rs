#[cfg(not(target_feature = "avx2"))]
compile_error!("This library requires AVX2 CPU feature.");

mod integer_256;
pub use integer_256::*;
