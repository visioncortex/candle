//! The decode fast path: CUDA's `fast_mmvq`, run on ROCm.
//!
//! Dots a batch of one to eight activation rows against the packed weights
//! with `candle-kernels/src/mmvq_gguf.cu`, the same source `quantized/
//! fast_mmvq.rs` launches on CUDA. It:
//!
//! * quantizes f32, f16 and bf16 activations to `q8_1` directly, with no cast
//!   to f32 first;
//! * writes the output in the *activation* dtype, so a bf16 model does not
//!   pay a cast back after each of a layer's seven matmuls;
//! * allocates both buffers uninitialised. The quantizer writes every byte of
//!   the padded `q8_1` rows and the matvec writes every `dst` element, so a
//!   memset of either would be dead work.
//!
//! Unlike CUDA's static, grow-only workspace, the `q8_1` scratch is a per-call
//! allocation. The device allocator makes that a host-side free-list pop, and
//! it is what routes through the `VmmPool` during HIP graph capture; a static
//! buffer grown mid-capture would not.
//!
//! The kernels are launched through `SharedLibUnit::Mmvq`, a shared library
//! rather than a `--genco` module, because the launchers are `extern "C"`
//! host functions that choose the batch-size instantiation themselves.
//!
//! No architecture gate: the kernels reduce with a 32-wide shuffle and launch
//! 32-wide blocks, which is one wavefront on RDNA. On CDNA's wave64 that
//! assumption is a constraint, as it is for the DMMV kernels.

use super::kernels::{launch_err, MATRIX_ROW_PADDING};
use super::q8_1::{buffer_bytes, force_dmmv, pad};
use super::QRocmStorage;
use crate::quantized::GgmlDType;
use crate::rocm_backend::{RocmStorage, RocmStorageSlice, SendSyncDeviceMemory, WrapErr};
use crate::{Layout, Result};
use candle_rocm_kernels::SharedLibUnit;
use std::ffi::c_void;

/// Largest batch `mmvq_gguf.cu`'s launcher has an instantiation for.
pub(super) const MAX_BATCH: usize = 8;

/// `QK8_1`: activation values per `q8_1` block.
const Q8_1_BLOCK: i32 = 32;

/// The fast path for [`super::QRocmStorage::fwd`], or `None` when the shape,
/// dtype or activation is one it does not cover and the caller should fall
/// through to the paths behind it.
///
/// `n` is the number of output rows, `k` the shared dimension and `b_size` the
/// flattened batch (`b * m`). The result holds `b_size * n` values in the
/// activation's own dtype, which is the contiguous `(b_size, n)` output the
/// caller wants.
pub(super) fn try_fwd(
    q: &QRocmStorage,
    n: usize,
    k: usize,
    b_size: usize,
    storage: &RocmStorage,
    layout: &Layout,
) -> Result<Option<RocmStorage>> {
    if force_dmmv() || !(1..=MAX_BATCH).contains(&b_size) {
        return Ok(None);
    }
    // Every activation type launches the same set of dtypes.
    if f32::matvec(q.dtype).is_none() || n == 0 || k == 0 || !k.is_multiple_of(q.dtype.block_size())
    {
        return Ok(None);
    }
    // The launcher's ABI is `int`; decline rather than truncate.
    let k_padded = pad(k, MATRIX_ROW_PADDING);
    if [n, k_padded, b_size]
        .iter()
        .any(|&v| i32::try_from(v).is_err())
    {
        return Ok(None);
    }
    let Some((o1, o2)) = layout.contiguous_offsets() else {
        return Ok(None);
    };
    if o2 - o1 != b_size * k {
        return Ok(None);
    }

    let slice = match &storage.slice {
        RocmStorageSlice::F32(y) => {
            // DMMV takes exactly a batch of one, contiguous and f32, and is
            // the better kernel for it below the size `dmmv_is_faster` knows.
            if b_size == 1
                && super::dmmv::has_kernel(q.dtype)
                && super::dmmv::dmmv_is_faster(q.dtype, n * k)
            {
                return Ok(None);
            }
            RocmStorageSlice::F32(mul_mat_vec(q, y, o1, (k, n), b_size)?)
        }
        RocmStorageSlice::F16(y) => RocmStorageSlice::F16(mul_mat_vec(q, y, o1, (k, n), b_size)?),
        RocmStorageSlice::BF16(y) => RocmStorageSlice::BF16(mul_mat_vec(q, y, o1, (k, n), b_size)?),
        _ => return Ok(None),
    };
    Ok(Some(RocmStorage {
        slice,
        device: q.device.clone(),
    }))
}

/// `y` (starting at element `y_offset`, `b_size` rows of `k`) against `q`'s
/// `(n, k)` weights, in `y`'s element type `T`, which picks the launchers
/// through [`Activation`]. `(k, n)` is the matrix shape.
///
/// ```text
/// launch_mmvq_gguf_quantize_q8_1_<act>(const void *x, void *vy,
///     int kx, int kx_padded, int num_rows, void *stream)
/// launch_mmvq_gguf_<tag>_<act>_plain(const void *vx, const void *vy,
///     void *dst, int ncols_x, int nrows_x, int stride_col_y,
///     int stride_col_dst, int b_size, void *stream)
/// ```
///
/// `stride_col_y` is the `q8_1` row stride in *blocks*, over the padded width
/// the quantizer lays rows out at; `stride_col_dst` is `n`.
///
/// Errors if `q`'s dtype has no launcher, if `q` holds fewer than `n * k`
/// elements, or if the shape overflows the launchers' `int` arguments. Callers
/// other than [`try_fwd`] rely on those checks, so they stay here.
pub(super) fn mul_mat_vec<T: Activation>(
    q: &QRocmStorage,
    y: &SendSyncDeviceMemory<T>,
    y_offset: usize,
    (k, n): (usize, usize),
    b_size: usize,
) -> Result<SendSyncDeviceMemory<T>> {
    let dtype = q.dtype;
    let Some(matvec_name) = T::matvec(dtype) else {
        crate::bail!("no mmvq_gguf kernel for {dtype:?}")
    };
    let data_elems = q.len / dtype.type_size() * dtype.block_size();
    if data_elems < k * n {
        crate::bail!("quantized mmvq_gguf: data holds {data_elems} elems, need {k}x{n}")
    }
    let (Ok(k_i), Ok(k_padded_i), Ok(n_i), Ok(b_i)) = (
        i32::try_from(k),
        i32::try_from(pad(k, MATRIX_ROW_PADDING)),
        i32::try_from(n),
        i32::try_from(b_size),
    ) else {
        crate::bail!("quantized mmvq_gguf: shape {b_size}x{k}x{n} overflows the launcher's int")
    };

    let dev = &q.device;
    // Uninitialised: see the module docs.
    let y_q8_1 = dev.alloc::<u8>(buffer_bytes(k, b_size))?;
    let dst = dev.alloc::<T>(n * b_size)?;
    let lib = dev.shared_library(SharedLibUnit::Mmvq)?;
    // SAFETY: both symbols name `extern "C"` launchers in the compiled
    // `mmvq_gguf` library whose signatures are exactly `QuantizeFn` and
    // `PlainFn`, for activations of type `T`; see `Activation` and
    // `candle-kernels/src/mmvq_gguf.cu`.
    let quantize: libloading::Symbol<QuantizeFn> =
        unsafe { lib.get(T::QUANTIZE.as_bytes()) }.w()?;
    let matvec: libloading::Symbol<PlainFn> = unsafe { lib.get(matvec_name.as_bytes()) }.w()?;

    // The launchers return `void`, so `hipGetLastError` is the only report of
    // a failed launch. It is sticky and global to the thread: clear whatever
    // an earlier, unchecked launch left so a failure below is this call's.
    let _ = crate::rocm_backend::rocm_rs::hip::get_last_error();

    let stream = dev.stream().as_raw().cast();
    // SAFETY: `y` holds `b_size * k` elements from `y_offset` (checked by the
    // caller against the layout), `y_q8_1` is sized for `b_size` padded rows,
    // `q.data` holds at least `n * k` elements, `dst` holds `n * b_size`, and
    // `stream` is this device's own HIP stream.
    unsafe {
        quantize(
            y.ptr_at(y_offset).cast_const(),
            y_q8_1.as_ptr(),
            k_i,
            k_padded_i,
            b_i,
            stream,
        );
    }
    // Returning here drops `y_q8_1` while the launch may still be queued. That
    // is sound: the allocator hands a buffer out again only to work on this
    // same stream, which runs behind it.
    check_launch(T::QUANTIZE)?;
    // SAFETY: as above.
    unsafe {
        matvec(
            q.data.as_ptr().cast_const(),
            y_q8_1.as_ptr().cast_const(),
            dst.as_ptr(),
            k_i,
            n_i,
            k_padded_i / Q8_1_BLOCK,
            n_i,
            b_i,
            stream,
        );
    }
    check_launch(matvec_name)?;
    Ok(dst)
}

/// `launch_mmvq_gguf_quantize_q8_1_<act>`: quantize `num_rows` activation rows
/// of `kx` elements to `q8_1`, laid out at `kx_padded` elements per row.
type QuantizeFn = unsafe extern "C" fn(
    x: *const c_void,
    vy: *mut c_void,
    kx: i32,
    kx_padded: i32,
    num_rows: i32,
    stream: *mut c_void,
);

/// `launch_mmvq_gguf_<tag>_<act>_plain`: dot `b_size` quantized rows against
/// the packed `(nrows_x, ncols_x)` weights into `dst`.
type PlainFn = unsafe extern "C" fn(
    vx: *const c_void,
    vy: *const c_void,
    dst: *mut c_void,
    ncols_x: i32,
    nrows_x: i32,
    stride_col_y: i32,
    stride_col_dst: i32,
    b_size: i32,
    stream: *mut c_void,
);

/// Surface a launch error the `<<<>>>` host launcher has no way to return.
/// `symbol` is the launcher's NUL-terminated name, used only to report it.
fn check_launch(symbol: &str) -> Result<()> {
    let err = crate::rocm_backend::rocm_rs::hip::get_last_error();
    if err.is_success() {
        Ok(())
    } else {
        Err(launch_err(symbol.trim_end_matches('\0'), err))
    }
}

///// Activation types ////////////////////////////////////////////////////

/// An activation element type `mmvq_gguf.cu` has launchers for, and the
/// launchers' names.
///
/// This is what ties a buffer's element type to the symbols that read it: a
/// launcher for another type would read the wrong width out of the buffer.
///
/// # Safety
/// Both names must be NUL-terminated symbols in `mmvq_gguf.cu`, with the
/// signatures `QuantizeFn` and `PlainFn`, whose activation and output
/// elements are `Self`.
pub(super) unsafe trait Activation {
    /// The launcher that quantizes rows of `Self` to `q8_1`.
    const QUANTIZE: &'static str;

    /// The launcher that dots `q8_1` rows against `dtype` weights and writes
    /// `Self`, or `None` if `mmvq_gguf.cu` has none for `dtype`.
    ///
    /// All lowercase, unlike `quantized.cu`'s `q4_K`-style spelling.
    fn matvec(dtype: GgmlDType) -> Option<&'static str>;
}

/// The `Activation` impl for one element type, with `$act` its symbol suffix.
macro_rules! activation {
    ($ty:ty, $act:literal) => {
        // SAFETY: the names are the `MMVQ_LAUNCHER_PLAIN(<tag>, $act, ..)` and
        // `launch_mmvq_gguf_quantize_q8_1_$act` launchers; the source-reading
        // test `launcher_names_exist_in_the_source` checks every one.
        unsafe impl Activation for $ty {
            const QUANTIZE: &'static str = concat!("launch_mmvq_gguf_quantize_q8_1_", $act, "\0");

            fn matvec(dtype: GgmlDType) -> Option<&'static str> {
                Some(match dtype {
                    GgmlDType::Q4_0 => concat!("launch_mmvq_gguf_q4_0_", $act, "_plain\0"),
                    GgmlDType::Q4_1 => concat!("launch_mmvq_gguf_q4_1_", $act, "_plain\0"),
                    GgmlDType::Q5_0 => concat!("launch_mmvq_gguf_q5_0_", $act, "_plain\0"),
                    GgmlDType::Q5_1 => concat!("launch_mmvq_gguf_q5_1_", $act, "_plain\0"),
                    GgmlDType::Q8_0 => concat!("launch_mmvq_gguf_q8_0_", $act, "_plain\0"),
                    GgmlDType::Q2K => concat!("launch_mmvq_gguf_q2_k_", $act, "_plain\0"),
                    GgmlDType::Q3K => concat!("launch_mmvq_gguf_q3_k_", $act, "_plain\0"),
                    GgmlDType::Q4K => concat!("launch_mmvq_gguf_q4_k_", $act, "_plain\0"),
                    GgmlDType::Q5K => concat!("launch_mmvq_gguf_q5_k_", $act, "_plain\0"),
                    GgmlDType::Q6K => concat!("launch_mmvq_gguf_q6_k_", $act, "_plain\0"),
                    _ => return None,
                })
            }
        }
    };
}

activation!(f32, "f32");
activation!(half::f16, "f16");
activation!(half::bf16, "bf16");
