//! The fused dequantize-and-GEMV path (`dequantize_mul_mat_vec_*`).
//!
//! Multiplies a single f32 activation vector straight against the packed
//! weights, so nothing is ever materialised in f32. [`super::fast_mmvq`] covers
//! batches of up to eight and is faster from [`dmmv_is_faster`]'s thresholds
//! up, so this runs below them or when `CANDLE_ROCM_FORCE_DMMV` is set — but
//! it is the more accurate of the two, since it never requantizes the
//! activations.

use super::kernels::{arg, launch_err, WARP_SIZE};
use crate::quantized::GgmlDType;
use crate::rocm_backend::rocm_rs::hip::Dim3;
use crate::rocm_backend::{kernels, RocmDevice, SendSyncDeviceMemory};
use crate::Result;

/// Name of the fused dequantize+GEMV (DMMV) kernel for `dtype`, or an error for
/// the dtypes `quantized.cu` ships no `dequantize_mul_mat_vec_*` for.
///
/// Mirrors `quantized/cuda.rs::dequantize_mul_mat_vec`'s name table exactly; the
/// K-quant kernels drop the `_cuda` suffix, matching the source.
fn kernel_name(dtype: GgmlDType) -> Result<&'static str> {
    let name = match dtype {
        GgmlDType::Q4_0 => "dequantize_mul_mat_vec_q4_0_cuda",
        GgmlDType::Q4_1 => "dequantize_mul_mat_vec_q4_1_cuda",
        GgmlDType::Q5_0 => "dequantize_mul_mat_vec_q5_0_cuda",
        GgmlDType::Q5_1 => "dequantize_mul_mat_vec_q5_1_cuda",
        GgmlDType::Q8_0 => "dequantize_mul_mat_vec_q8_0_cuda",
        GgmlDType::Q2K => "dequantize_mul_mat_vec_q2_k",
        GgmlDType::Q3K => "dequantize_mul_mat_vec_q3_k",
        GgmlDType::Q4K => "dequantize_mul_mat_vec_q4_k",
        GgmlDType::Q5K => "dequantize_mul_mat_vec_q5_k",
        GgmlDType::Q6K => "dequantize_mul_mat_vec_q6_k",
        _ => crate::bail!("no ROCm dequantize_mul_mat_vec kernel for {dtype:?}"),
    };
    Ok(name)
}

/// True when `dtype` has a fused dequantize+GEMV (DMMV) kernel.
pub(super) fn has_kernel(dtype: GgmlDType) -> bool {
    kernel_name(dtype).is_ok()
}

/// Weight elements below which a `q8_1` decode kernel's fixed cost outweighs
/// what it saves, for the dtypes where it is ever worth paying at all. See
/// [`dmmv_is_faster`].
const Q8_1_MIN_WORK: usize = 8 << 20;

/// Whether a batch of one f32 row is better served by this module than by
/// [`super::fast_mmvq`].
///
/// The `q8_1` path pays for a second kernel launch and a second allocation to
/// build the `q8_1` activation, roughly 14 µs of fixed cost on gfx1101, and
/// only earns it back on a large enough weight matrix. Measured at `m = 1`,
/// f32 activations, milliseconds per matmul, DMMV against the `q8_1` decode
/// kernels of `quantized.cu` (`mul_mat_vec_*_q8_1_cuda1`), which this path's
/// kernels replaced:
///
/// ```text
///  k     n       q8_0           q4_0           q4_K           q6_K
/// 2048  2048  0.0158 0.0179  0.0121 0.0161  0.0070 0.0181  0.0091 0.0196
/// 2048  4096  0.0227 0.0226  0.0226 0.0207  0.0108 0.0238  0.0153 0.0261
/// 4096  4096  0.0429 0.0261  0.0433 0.0222  0.0174 0.0329  0.0280 0.0370
/// 4096 32000  0.3716 0.2495  0.3487 0.1423  0.1572 0.1637  0.2235 0.2026
/// ```
///
/// Two things fall out. The legacy quants have a much weaker DMMV kernel than
/// the K-quants do, so the `q8_1` path overtakes them at roughly 8 Mi weight
/// elements and is then up to 2.4x faster. The K-quant DMMV kernels are
/// already good enough that it never clearly wins — 10% at the 131 Mi
/// `lm_head`, less or nothing everywhere else — so they keep DMMV at every
/// size.
///
/// None of this applies above a batch of one, or to an f16/bf16 activation:
/// there DMMV is not an option at all and the alternative is the dense
/// dequantize, which a `q8_1` kernel beats by an order of magnitude.
pub(super) fn dmmv_is_faster(dtype: GgmlDType, work: usize) -> bool {
    match dtype {
        GgmlDType::Q2K | GgmlDType::Q3K | GgmlDType::Q4K | GgmlDType::Q5K | GgmlDType::Q6K => true,
        _ => work < Q8_1_MIN_WORK,
    }
}

/// Output rows per thread block (`GGML_CUDA_MMV_Y` in `cuda.rs`).
///
/// Two was tried, on the theory that one wave32 per workgroup leaves the RDNA3
/// scheduler nothing to interleave against a memory stall (llama.cpp uses two
/// for the K-quant DMMV). On gfx1101 it measured neutral to 7% *worse* —
/// `4096x32000` q4_K went 0.155 ms to 0.167 ms, `4096x4096` q4_K 0.0178 to
/// 0.0188, q4_0 and q6_K unchanged, over two runs each. These kernels are
/// bandwidth bound, not occupancy bound, so the second wave buys nothing and
/// costs a wider grid tail. Kept at one.
///
/// Anyone raising it must handle three different bound checks, because the
/// kernels do not agree on theirs:
///
/// * generic template (`quantized.cu:1473`, used by q4_0/q4_1/q5_0/q5_1/q8_0) —
///   `if (row >= nrows) return;`. Correct for any `nrows`.
/// * q2_k/q3_k/q4_k/q6_k (`quantized.cu:1559`, `:1663`, `:1767`, `:2021`) —
///   `if (row > nrows) return;`, with `>` rather than `>=`. `row == nrows`
///   passes the check and writes `dst[nrows]`, one element past the output.
///   At one row per block the grid is exactly `nrows` blocks so that row is
///   unreachable; at two with an *odd* `nrows` it is the last block's second
///   wave.
/// * q5_k (`quantized.cu:1900`) — takes no `nrows` parameter at all and derives
///   `const int row = blockIdx.x;`, ignoring `threadIdx.y`. A second wave would
///   recompute and rewrite the same row while the upper half of the matrix went
///   uncomputed, at any parity.
const MMV_Y: usize = 1;

/// Fused dequantize + matrix-vector product against the *packed* quantized
/// weights.
///
/// `data` is the `(nrows, ncols)` quantized weight matrix (padded), `y` a single
/// dense `ncols`-element f32 activation vector starting at element `y_offset`.
/// The result is `nrows` f32 dot products. `data_len` is the *logical* payload
/// length of `data` in bytes (the buffer itself carries `MATRIX_ROW_PADDING`
/// extra elements).
///
/// ```text
/// dequantize_mul_mat_vec_*(const void *vx, const dfloat/float *y,
///                          float *dst, const int ncols, const int nrows)
/// ```
///
/// The `q5_k` kernel omits the trailing `nrows` parameter; passing it anyway is
/// harmless — the kernel simply never reads that slot — and keeps one arg shape
/// for every dtype, exactly as `cuda.rs` does.
#[allow(clippy::too_many_arguments)]
pub(super) fn mul_mat_vec(
    data: &SendSyncDeviceMemory<u8>,
    data_len: usize,
    y: &SendSyncDeviceMemory<f32>,
    y_offset: usize,
    dtype: GgmlDType,
    ncols: usize,
    nrows: usize,
    dev: &RocmDevice,
) -> Result<SendSyncDeviceMemory<f32>> {
    let data_elems = data_len / dtype.type_size() * dtype.block_size();
    if data_elems < ncols * nrows {
        crate::bail!("quantized dmmv: data holds {data_elems} elems, need {ncols}x{nrows}")
    }
    let name = kernel_name(dtype)?;
    let func = dev.get_or_load_func(name, &kernels::QUANTIZED)?;
    let dst = dev.alloc::<f32>(nrows)?;
    if nrows == 0 || ncols == 0 {
        return Ok(dst);
    }
    let block_num_y = nrows.div_ceil(MMV_Y);
    let src_ptr = data.as_ptr();
    let y_ptr = unsafe { y.ptr_at(y_offset) };
    let dst_ptr = dst.as_ptr();
    let ncols_i = ncols as i32;
    let nrows_i = nrows as i32;
    let mut args = vec![
        arg(&src_ptr),
        arg(&y_ptr),
        arg(&dst_ptr),
        arg(&ncols_i),
        arg(&nrows_i),
    ];
    func.launch(
        Dim3::new_1d(block_num_y as u32),
        Dim3::new_2d(WARP_SIZE as u32, MMV_Y as u32),
        0,
        Some(dev.stream()),
        &mut args,
    )
    .map_err(|e| launch_err(name, e))?;
    Ok(dst)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The DMMV kernel names must match `quantized.cu` byte for byte — the
    /// K-quants drop the `_cuda` suffix and `has_kernel` must agree with which
    /// dtypes have a kernel at all.
    #[test]
    fn dmmv_kernel_names() -> Result<()> {
        assert_eq!(
            kernel_name(GgmlDType::Q4_0)?,
            "dequantize_mul_mat_vec_q4_0_cuda"
        );
        assert_eq!(kernel_name(GgmlDType::Q4K)?, "dequantize_mul_mat_vec_q4_k");
        assert_eq!(kernel_name(GgmlDType::Q6K)?, "dequantize_mul_mat_vec_q6_k");
        assert!(has_kernel(GgmlDType::Q4K));
        // No dmmv kernel is compiled for Q8K or the float dtypes.
        assert!(!has_kernel(GgmlDType::Q8K));
        assert!(kernel_name(GgmlDType::Q8K).is_err());
        Ok(())
    }

    /// At a batch of one, DMMV is the cheaper path for the K-quants at any
    /// size and for the legacy quants below [`Q8_1_MIN_WORK`]; see
    /// [`dmmv_is_faster`] for the numbers.
    #[test]
    fn dmmv_wins_the_small_single_row_shapes() {
        // SmolLM2-360M's projections: far below the threshold.
        assert!(dmmv_is_faster(GgmlDType::Q8_0, 960 * 2560));
        assert!(dmmv_is_faster(GgmlDType::Q4_0, 2560 * 960));
        // A 7B-shaped projection and lm_head: `q8_1`, for the legacy quants
        // only.
        assert!(!dmmv_is_faster(GgmlDType::Q4_0, 4096 * 4096));
        assert!(!dmmv_is_faster(GgmlDType::Q8_0, 4096 * 32000));
        // The K-quant DMMV kernels are good enough to keep at every size.
        assert!(dmmv_is_faster(GgmlDType::Q4K, 4096 * 32000));
        assert!(dmmv_is_faster(GgmlDType::Q6K, 4096 * 32000));
        assert!(dmmv_is_faster(GgmlDType::Q2K, 960 * 960));
    }
}
