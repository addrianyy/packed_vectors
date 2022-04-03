## Example:
```rs
fn mandelbrot_escape(cr: Float64x4, ci: Float64x4, max_iters: u32) -> Uint64x4 {
    let two = Float64x4::splat(2.0);
    let four = Float64x4::splat(4.0);
    let one = Uint64x4::splat(1);

    let mut zr = cr;
    let mut zi = ci;

    let mut count = Uint64x4::zero();
    let mut alive_mask = Uint64x4::zero();

    for _ in 0..max_iters {
        let rsqr = zr * zr;
        let isqr = zi * zi;

        let alive_mask_f = (rsqr + isqr).lt(four);
        if alive_mask_f.mask() == 0 {
            return count;
        }

        alive_mask = alive_mask_f.transmute();
        count += one & alive_mask;

        // zi = ci + two * zr * zi;
        zi = two.fmadd(zr * zi, ci);
        zr = cr + rsqr - isqr;
    }

    // count & ~alive_mask
    alive_mask.andnot(count)
}
```