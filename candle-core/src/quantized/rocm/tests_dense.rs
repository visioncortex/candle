//! The dense fallback, and the dispatch that now prefers it at prefill.
//!
//! The reorientation is a layout change, so a mistake in it lands scrambled
//! values rather than drifted ones — hence a plain f32 matmul as the reference.
//! The dispatch is a policy change, so `preferred` is pinned on both sides of
//! its threshold and on the cases it must refuse.

use super::dense;
use crate::quantized::{GgmlDType, QMatMul, QStorage, QTensor};
use crate::rocm_backend::RocmDevice;
use crate::{DType, Device, Module, Result, Storage, Tensor};

macro_rules! rocm_device {
    () => {
        match RocmDevice::new(0) {
            Ok(dev) => Device::Rocm(dev),
            Err(_) => return Ok(()),
        }
    };
}

/// rocBLAS accumulates f16 in f32, but the operands carry ten bits of mantissa
/// and the sum runs over `k` terms. Relative to the row's largest magnitude,
/// for the reason `tests_mmq` gives.
const TOL: f32 = 1e-2;

fn assert_close(got: &[f32], want: &[f32], what: &str) {
    let scale = want.iter().fold(1.0f32, |acc, v| acc.max(v.abs()));
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (a - b).abs() <= TOL * scale,
            "{what}: element {i} was {a}, expected {b} (scale {scale})"
        );
    }
}

/// A gfx1101 (RDNA3) `amd_cc`, the generation `dense::preferred` prefers.
const RDNA3_CC: Option<i32> = Some(0x0100_0000 + 0x1101);
/// A gfx1201 (RDNA4) `amd_cc`, which must never take this path.
const RDNA4_CC: Option<i32> = Some(0x0100_0000 + 0x1201);

#[test]
fn preferred_only_where_the_sweep_says_so() {
    // Every batch/weight/dtype condition below is necessary but not
    // sufficient: this path only ever wins on RDNA3 (`dense::is_rdna3`), so
    // each qualifying case is checked against both an RDNA3 and a non-RDNA3
    // `amd_cc`, with no GPU required.
    let (n, k) = (2048usize, 2048usize);
    // The threshold is on the batch, and 256 is the first that qualifies.
    assert!(!dense::preferred(
        DType::F16,
        GgmlDType::Q8_0,
        n,
        k,
        255,
        RDNA3_CC
    ));
    assert!(dense::preferred(
        DType::F16,
        GgmlDType::Q8_0,
        n,
        k,
        256,
        RDNA3_CC
    ));
    assert!(!dense::preferred(
        DType::F16,
        GgmlDType::Q8_0,
        n,
        k,
        256,
        RDNA4_CC
    ));
    assert!(dense::preferred(
        DType::F16,
        GgmlDType::Q8_0,
        n,
        k,
        4096,
        RDNA3_CC
    ));
    assert!(!dense::preferred(
        DType::F16,
        GgmlDType::Q8_0,
        n,
        k,
        4096,
        RDNA4_CC
    ));

    // f32 and bf16 keep MMQ at every batch on every architecture: neither was
    // measured ahead of it, and bf16 has no dequantize kernel to reach this
    // path cheaply.
    assert!(!dense::preferred(
        DType::F32,
        GgmlDType::Q8_0,
        n,
        k,
        4096,
        RDNA3_CC
    ));
    assert!(!dense::preferred(
        DType::BF16,
        GgmlDType::Q8_0,
        n,
        k,
        4096,
        RDNA3_CC
    ));

    // A vocabulary-sized matrix would hold half a gigabyte of transient, on
    // every architecture.
    assert!(!dense::preferred(
        DType::F16,
        GgmlDType::Q8_0,
        32000,
        4096,
        4096,
        RDNA3_CC
    ));
    // ...while every projection in a model this card runs is under the cap.
    assert!(dense::preferred(
        DType::F16,
        GgmlDType::Q8_0,
        8192,
        2048,
        1024,
        RDNA3_CC
    ));
    assert!(!dense::preferred(
        DType::F16,
        GgmlDType::Q8_0,
        8192,
        2048,
        1024,
        RDNA4_CC
    ));

    // A dtype with no dequantize kernel would have to round-trip through the
    // host to reach this path, so it stays with MMQ and the fallback, on
    // every architecture.
    assert!(!dense::preferred(
        DType::F16,
        GgmlDType::Q8_1,
        n,
        k,
        4096,
        RDNA3_CC
    ));
}

/// The reoriented path against a plain f32 matmul of the same dequantized
/// weights — the test that the reorientation is the identity on the result.
///
/// Calls `dense::forward` directly rather than through `QMatMul::forward`:
/// `preferred` only routes the live dispatch here on RDNA3 (`dense::is_rdna3`),
/// but the reorientation itself must be correct wherever it runs.
#[test]
fn the_reoriented_dense_path_matches_a_dequantized_matmul() -> Result<()> {
    let device = rocm_device!();
    let m = 256usize;
    // Non-square, so a transposed result cannot pass by coincidence.
    for (k, n) in [(2048usize, 512usize), (512, 2048)] {
        let lhs: Vec<f32> = (0..m * k).map(|i| (i as f32 / 53.).sin()).collect();
        let rhs: Vec<f32> = (0..n * k).map(|i| (i as f32 / 71.).cos()).collect();
        let lhs = Tensor::from_slice(&lhs, (m, k), &device)?;
        let rhs = Tensor::from_slice(&rhs, (n, k), &device)?;

        for dtype in [GgmlDType::Q8_0, GgmlDType::Q4K, GgmlDType::Q4_0] {
            let qt = QTensor::quantize(&rhs, dtype)?;
            let want = lhs
                .matmul(&qt.dequantize(&device)?.t()?)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let QStorage::Rocm(q) = &qt.storage else {
                return Ok(());
            };
            let lhs16 = lhs.to_dtype(DType::F16)?;
            let (act, act_l) = lhs16.storage_and_layout();
            let Storage::Rocm(act) = &*act else {
                return Ok(());
            };
            let out = dense::forward(q, (1, m, n, k), act, act_l)?;
            let got = crate::tensor::from_storage(
                Storage::Rocm(out),
                (m, n),
                crate::op::BackpropOp::none(),
                false,
            )
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
            assert_close(&got, &want, &format!("{dtype:?} {k}x{n}"));
        }
    }
    Ok(())
}

/// A transformer hands the matmul `(b, m, k)` with the rhs broadcast over `b`,
/// which is where a zero batch stride reaches rocBLAS.
///
/// Calls `dense::forward` directly, for the same reason given in
/// `the_reoriented_dense_path_matches_a_dequantized_matmul`.
#[test]
fn the_reoriented_dense_path_handles_a_batched_activation() -> Result<()> {
    let device = rocm_device!();
    let (b, m, k, n) = (2usize, 192usize, 2048usize, 1024usize);
    let lhs: Vec<f32> = (0..b * m * k).map(|i| (i as f32 / 53.).sin()).collect();
    let rhs: Vec<f32> = (0..n * k).map(|i| (i as f32 / 71.).cos()).collect();
    let lhs = Tensor::from_slice(&lhs, (b, m, k), &device)?;
    let rhs = Tensor::from_slice(&rhs, (n, k), &device)?;

    let qt = QTensor::quantize(&rhs, GgmlDType::Q8_0)?;
    let want = lhs
        .matmul(&qt.dequantize(&device)?.t()?.broadcast_as((b, k, n))?)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let QStorage::Rocm(q) = &qt.storage else {
        return Ok(());
    };
    let lhs16 = lhs.to_dtype(DType::F16)?;
    let (act, act_l) = lhs16.storage_and_layout();
    let Storage::Rocm(act) = &*act else {
        return Ok(());
    };
    let out = dense::forward(q, (b, m, n, k), act, act_l)?;
    let got = crate::tensor::from_storage(
        Storage::Rocm(out),
        (b, m, n),
        crate::op::BackpropOp::none(),
        false,
    );
    assert_eq!(got.dims(), &[b, m, n]);
    let got = got.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    assert_close(&got, &want, "batched");
    Ok(())
}

/// Below the threshold the answer still has to come out of MMQ unchanged —
/// the case where a mistake in `preferred` silently changes which kernel ran.
#[test]
fn the_paths_below_the_threshold_still_agree() -> Result<()> {
    let device = rocm_device!();
    let (k, n) = (2048usize, 1024usize);
    let rhs: Vec<f32> = (0..n * k).map(|i| (i as f32 / 71.).cos()).collect();
    let rhs = Tensor::from_slice(&rhs, (n, k), &device)?;
    let qt = QTensor::quantize(&rhs, GgmlDType::Q8_0)?;
    let deq = qt.dequantize(&device)?.t()?;
    let mm = QMatMul::from_qtensor(qt)?;

    for m in [1usize, 64, 255, 256] {
        let lhs: Vec<f32> = (0..m * k).map(|i| (i as f32 / 53.).sin()).collect();
        let lhs = Tensor::from_slice(&lhs, (m, k), &device)?;
        let want = lhs.matmul(&deq)?.flatten_all()?.to_vec1::<f32>()?;
        let got = mm
            .forward(&lhs.to_dtype(DType::F16)?)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        // MMQ requantizes the activations to q8_1, which is a coarser
        // computation than the f16 GEMM above it; hold it to `tests_mmq`'s 2%.
        let scale = want.iter().fold(1.0f32, |acc, v| acc.max(v.abs()));
        for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
            assert!(
                (a - b).abs() <= 2e-2 * scale,
                "m={m}: element {i} was {a}, expected {b}"
            );
        }
    }
    Ok(())
}
