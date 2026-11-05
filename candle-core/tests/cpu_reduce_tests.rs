//! The backward pass's CPU paths run in parallel: a large sum split over
//! destinations (ReduceSum), a large affine split into blocks, the transpose
//! convolution's NHWC packing split over rows, and conv2d's flattened kernel
//! (as large as an activation when the kernel is an output gradient) split
//! over output channels. These pin the parallel paths — reached only above a
//! size threshold, so the shapes here are deliberately large — against the
//! serial algorithm, bit for bit: a sum's destination must receive its
//! sources in increasing source order, so the float sums are the same at any
//! thread count, and a copy must be a copy.
use candle_core::{Device, Result, Tensor};

/// Values of mixed magnitude, so a change in summation order shows in the bits.
fn noise(n: usize) -> Vec<f32> {
    let mut s = 0x9e37_79b9_7f4a_7c15u64;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let u = (s >> 40) as f32 / (1u64 << 24) as f32;
            (u - 0.5) * [1e-3, 1.0, 1e3][(s & 3) as usize % 3]
        })
        .collect()
}

/// The serial reduce: walk the source in order, add each element into the
/// destination its kept indices name.
fn naive_sum(src: &[f32], dims: &[usize], reduce: &[usize]) -> Vec<f32> {
    let dst_dims: Vec<usize> = dims
        .iter()
        .enumerate()
        .map(|(d, &s)| if reduce.contains(&d) { 1 } else { s })
        .collect();
    let mut dst = vec![0f32; dst_dims.iter().product()];
    let mut idx = vec![0usize; dims.len()];
    for &v in src {
        let mut o = 0;
        for d in 0..dims.len() {
            o = o * dst_dims[d] + if reduce.contains(&d) { 0 } else { idx[d] };
        }
        dst[o] += v;
        for d in (0..dims.len()).rev() {
            idx[d] += 1;
            if idx[d] < dims[d] {
                break;
            }
            idx[d] = 0;
        }
    }
    dst
}

fn check_sum(dims: &[usize], reduce: &[usize]) -> Result<()> {
    let n: usize = dims.iter().product();
    let v = noise(n);
    let t = Tensor::from_vec(v.clone(), dims, &Device::Cpu)?;
    let got = t.sum_keepdim(reduce)?.flatten_all()?.to_vec1::<f32>()?;
    let want = naive_sum(&v, dims, reduce);
    assert_eq!(got.len(), want.len(), "{dims:?} over {reduce:?}");
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(
            g.to_bits(),
            w.to_bits(),
            "{dims:?} over {reduce:?}: dst {i} {g} vs {w}"
        );
    }
    Ok(())
}

#[test]
fn large_bias_grad_sum_is_bit_identical() -> Result<()> {
    // a conv bias gradient: inner group reduced, an outer reduced group
    check_sum(&[4, 48, 64, 64], &[0, 2, 3])
}

#[test]
fn large_sum_over_leading_dim_is_bit_identical() -> Result<()> {
    // inner group kept: destinations in one long run, split into slices
    check_sum(&[64, 8192], &[0])
}

#[test]
fn large_sum_over_middle_dim_is_bit_identical() -> Result<()> {
    check_sum(&[8, 64, 1024], &[1])
}

#[test]
fn large_sum_alternating_groups_is_bit_identical() -> Result<()> {
    // reduced, kept, reduced, kept
    check_sum(&[16, 3, 128, 64], &[0, 2])
}

#[test]
fn large_sum_with_unit_dims_is_bit_identical() -> Result<()> {
    check_sum(&[1, 48, 1, 128, 64], &[0, 2, 3, 4])
}

#[test]
fn large_sum_over_last_dims_is_bit_identical() -> Result<()> {
    // the trailing-dims fast path, one destination per task
    let dims = [512, 1024];
    let v = noise(dims.iter().product());
    let t = Tensor::from_vec(v.clone(), &dims, &Device::Cpu)?;
    let got = t.sum_keepdim(1)?.flatten_all()?.to_vec1::<f32>()?;
    // the fast path is vec_reduce_sum per row; the same call on a row that is
    // below the threshold on its own is the serial reference
    for r in 0..dims[0] {
        let row = Tensor::from_vec(
            v[r * dims[1]..(r + 1) * dims[1]].to_vec(),
            (1, dims[1]),
            &Device::Cpu,
        )?;
        let w = row.sum_keepdim(1)?.flatten_all()?.to_vec1::<f32>()?[0];
        assert_eq!(got[r].to_bits(), w.to_bits(), "row {r}");
    }
    Ok(())
}

#[test]
fn large_affine_matches_naive() -> Result<()> {
    // contiguous (the parallel block) and a transposed view (the serial map)
    let (h, w) = (700, 900);
    let v = noise(h * w);
    let t = Tensor::from_vec(v.clone(), (h, w), &Device::Cpu)?;
    let (mul, add) = (0.37f64, -1.25f64);
    let got = t.affine(mul, add)?.flatten_all()?.to_vec1::<f32>()?;
    let want: Vec<f32> = v.iter().map(|&x| x * mul as f32 + add as f32).collect();
    assert!(got
        .iter()
        .zip(&want)
        .all(|(a, b)| a.to_bits() == b.to_bits()));
    let got_t = t
        .t()?
        .affine(mul, add)?
        .t()?
        .flatten_all()?
        .to_vec1::<f32>()?;
    assert!(got_t
        .iter()
        .zip(&want)
        .all(|(a, b)| a.to_bits() == b.to_bits()));
    Ok(())
}

#[test]
fn large_conv_transpose2d_matches_small_batches() -> Result<()> {
    // (8, 48, 64, 64) is above the threshold, one batch (196k) below it: the
    // packing differs, the per-pixel dot products must not
    let (b, c_in, c_out, h, w) = (8, 48, 24, 64, 64);
    let x = Tensor::from_vec(noise(b * c_in * h * w), (b, c_in, h, w), &Device::Cpu)?;
    let k = Tensor::from_vec(noise(c_in * c_out * 9), (c_in, c_out, 3, 3), &Device::Cpu)?;
    for x in [x.clone(), x.transpose(2, 3)?] {
        let got = x.conv_transpose2d(&k, 1, 0, 1, 1)?;
        let parts = (0..b)
            .map(|i| {
                x.narrow(0, i, 1)?
                    .contiguous()?
                    .conv_transpose2d(&k, 1, 0, 1, 1)
            })
            .collect::<Result<Vec<_>>>()?;
        let want = Tensor::cat(&parts, 0)?;
        let (g, w) = (
            got.flatten_all()?.to_vec1::<f32>()?,
            want.flatten_all()?.to_vec1::<f32>()?,
        );
        assert!(g.iter().zip(&w).all(|(a, b)| a.to_bits() == b.to_bits()));
    }
    Ok(())
}

#[test]
fn large_kernel_conv2d_matches_small_kernel_groups() -> Result<()> {
    // The kernel gradient's shape: the "kernel" is an output gradient
    // transposed to (c_out, batch, h, w) — strided, and as large as an
    // activation. 64·8·32·32 = 524k is above the threshold; a group of 16
    // output channels (131k) is below it.
    let (b, c_in, c_out, h, w, kh) = (2, 8, 64, 48, 48, 32);
    let x = Tensor::from_vec(noise(b * c_in * h * w), (b, c_in, h, w), &Device::Cpu)?;
    let g = Tensor::from_vec(
        noise(c_in * c_out * kh * kh),
        (c_in, c_out, kh, kh),
        &Device::Cpu,
    )?;
    let k = g.transpose(0, 1)?; // (c_out, c_in, kh, kh), non-contiguous
    let got = x.conv2d(&k, 0, 1, 1, 1)?;
    let parts = (0..c_out / 16)
        .map(|i| x.conv2d(&k.narrow(0, i * 16, 16)?, 0, 1, 1, 1))
        .collect::<Result<Vec<_>>>()?;
    let want = Tensor::cat(&parts, 1)?;
    let (g, w) = (
        got.flatten_all()?.to_vec1::<f32>()?,
        want.flatten_all()?.to_vec1::<f32>()?,
    );
    assert_eq!(g.len(), w.len());
    assert!(g.iter().zip(&w).all(|(a, b)| a.to_bits() == b.to_bits()));
    Ok(())
}
