//! The CPU backend fans large copies out over rayon (copy2d, strided-source
//! copies, nearest upsample, the conv2d NHWC transpose). These pin the
//! parallel paths — reached only above a size threshold, so the shapes here
//! are deliberately large — against naive index arithmetic.
use candle_core::{DType, Device, Result, Tensor};

fn ramp(dims: (usize, usize, usize, usize)) -> Result<Tensor> {
    let (b, c, h, w) = dims;
    let n = b * c * h * w;
    let v: Vec<f32> = (0..n).map(|i| (i % 7919) as f32 * 0.5 - 3.0).collect();
    Tensor::from_vec(v, dims, &Device::Cpu)
}

#[test]
fn large_transpose_contiguous_matches_naive() -> Result<()> {
    let (b, c, h, w) = (2, 3, 640, 700);
    let t = ramp((b, c, h, w))?;
    let got = t
        .transpose(2, 3)?
        .contiguous()?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let src = t.flatten_all()?.to_vec1::<f32>()?;
    let mut want = vec![0f32; src.len()];
    for bi in 0..b {
        for ci in 0..c {
            for hi in 0..h {
                for wi in 0..w {
                    want[((bi * c + ci) * w + wi) * h + hi] =
                        src[((bi * c + ci) * h + hi) * w + wi];
                }
            }
        }
    }
    assert_eq!(got, want);
    Ok(())
}

#[test]
fn large_narrow_contiguous_matches_naive() -> Result<()> {
    let (b, c, h, w) = (1, 5, 900, 1000);
    let t = ramp((b, c, h, w))?;
    let (y0, nh, x0, nw) = (17, 777, 23, 901);
    let got = t
        .narrow(2, y0, nh)?
        .narrow(3, x0, nw)?
        .contiguous()?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let src = t.flatten_all()?.to_vec1::<f32>()?;
    let mut want = Vec::with_capacity(c * nh * nw);
    for ci in 0..c {
        for hi in 0..nh {
            for wi in 0..nw {
                want.push(src[(ci * h + y0 + hi) * w + x0 + wi]);
            }
        }
    }
    assert_eq!(got, want);
    Ok(())
}

#[test]
fn large_cat_matches_naive() -> Result<()> {
    let (b, c, h) = (1, 4, 800);
    let (wa, wb) = (600, 450);
    let a = ramp((b, c, h, wa))?;
    let bt = (ramp((b, c, h, wb))? * 2.0)?;
    let got = Tensor::cat(&[&a, &bt], 3)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let sa = a.flatten_all()?.to_vec1::<f32>()?;
    let sb = bt.flatten_all()?.to_vec1::<f32>()?;
    let mut want = Vec::with_capacity(c * h * (wa + wb));
    for ci in 0..c {
        for hi in 0..h {
            want.extend_from_slice(&sa[(ci * h + hi) * wa..(ci * h + hi + 1) * wa]);
            want.extend_from_slice(&sb[(ci * h + hi) * wb..(ci * h + hi + 1) * wb]);
        }
    }
    assert_eq!(got, want);
    // Along the row axis the pieces are whole blocks: a single memcpy each.
    let got2 = Tensor::cat(&[&a, &a], 2)?.flatten_all()?.to_vec1::<f32>()?;
    let mut want2 = Vec::with_capacity(2 * c * h * wa);
    for ci in 0..c {
        want2.extend_from_slice(&sa[ci * h * wa..(ci + 1) * h * wa]);
        want2.extend_from_slice(&sa[ci * h * wa..(ci + 1) * h * wa]);
    }
    assert_eq!(got2, want2);
    Ok(())
}

#[test]
fn large_upsample_nearest2d_matches_naive() -> Result<()> {
    let (b, c, h, w) = (1, 3, 500, 640);
    let t = ramp((b, c, h, w))?;
    let (oh, ow) = (2 * h, 2 * w);
    let got = t
        .upsample_nearest2d(oh, ow)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let src = t.flatten_all()?.to_vec1::<f32>()?;
    let mut want = Vec::with_capacity(c * oh * ow);
    for ci in 0..c {
        for hi in 0..oh {
            for wi in 0..ow {
                want.push(src[(ci * h + hi / 2) * w + wi / 2]);
            }
        }
    }
    assert_eq!(got, want);
    // A strided (transposed) source takes the same path.
    let tt = t.transpose(2, 3)?;
    let got_t = tt
        .upsample_nearest2d(2 * w, 2 * h)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let want_t = tt
        .contiguous()?
        .upsample_nearest2d(2 * w, 2 * h)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    assert_eq!(got_t, want_t);
    Ok(())
}

#[test]
fn large_conv2d_matches_small_tiles() -> Result<()> {
    // The NHWC transpose inside conv2d_tiled is parallel per row; the conv
    // of a large image must equal the convs of its overlapping halves.
    let (c_in, c_out, h, w) = (8, 6, 300, 420);
    let x = ramp((1, c_in, h, w))?;
    let k = (Tensor::arange(0f32, (c_out * c_in * 9) as f32, &Device::Cpu)?
        .reshape((c_out, c_in, 3, 3))?
        * 1e-3)?;
    let whole = x.conv2d(&k, 1, 1, 1, 1)?;
    let top = x
        .narrow(2, 0, h / 2 + 1)?
        .conv2d(&k, 1, 1, 1, 1)?
        .narrow(2, 0, h / 2)?;
    let bot = x
        .narrow(2, h / 2 - 1, h / 2 + 1)?
        .conv2d(&k, 1, 1, 1, 1)?
        .narrow(2, 1, h / 2)?;
    let halves = Tensor::cat(&[&top, &bot], 2)?;
    let diff = (whole - halves)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff < 1e-4, "max |diff| {diff}");
    // And a 1x1 kernel, which takes the matmul fast path.
    let k1 = (Tensor::arange(0f32, (c_out * c_in) as f32, &Device::Cpu)?
        .reshape((c_out, c_in, 1, 1))?
        * 1e-2)?;
    let whole1 = x.conv2d(&k1, 0, 1, 1, 1)?;
    let left = x.narrow(3, 0, w / 2)?.conv2d(&k1, 0, 1, 1, 1)?;
    let right = x.narrow(3, w / 2, w - w / 2)?.conv2d(&k1, 0, 1, 1, 1)?;
    let diff1 = (whole1 - Tensor::cat(&[&left, &right], 3)?)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff1 < 1e-4, "max |diff| {diff1}");
    assert_eq!(x.dtype(), DType::F32);
    Ok(())
}
