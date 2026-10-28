//! Tests for the `mmvq_gguf` decode path.
//!
//! The first test needs no GPU: it pins the launcher names against the macros
//! in the shared source. The rest are hardware tests that skip without a
//! device. They compare against a dequantize-then-matmul reference with a
//! tolerance of 2% of the largest output: the activations are requantized to
//! `q8_1`, a numerically different computation from the reference.

use super::fast_mmvq::{self, Activation};
use crate::quantized::{GgmlDType, QMatMul, QStorage, QTensor};
use crate::rocm_backend::RocmDevice;
use crate::{DType, Device, Module, Result, Storage, Tensor};

/// `RocmDevice::new` fails on machines without a GPU; those runs skip.
macro_rules! rocm_device {
    () => {
        match RocmDevice::new(0) {
            Ok(dev) => Device::Rocm(dev),
            Err(_) => return Ok(()),
        }
    };
}

/// Every dtype `mmvq_gguf.cu` has launchers for.
const DTYPES: [GgmlDType; 10] = [
    GgmlDType::Q4_0,
    GgmlDType::Q4_1,
    GgmlDType::Q5_0,
    GgmlDType::Q5_1,
    GgmlDType::Q8_0,
    GgmlDType::Q2K,
    GgmlDType::Q3K,
    GgmlDType::Q4K,
    GgmlDType::Q5K,
    GgmlDType::Q6K,
];

/// Every launcher name an activation type resolves to exists in the shared
/// source. A misspelt one resolves to no symbol at run time.
#[test]
fn launcher_names_exist_in_the_source() {
    let source = include_str!("../../../../candle-kernels/src/mmvq_gguf.cu");
    check_launchers::<f32>(source, "f32");
    check_launchers::<half::f16>(source, "f16");
    check_launchers::<half::bf16>(source, "bf16");
    assert_eq!(f32::matvec(GgmlDType::Q8_1), None);
}

/// Check `T`'s launcher names against `source`, where `act` is the suffix the
/// names should carry.
fn check_launchers<T: Activation>(source: &str, act: &str) {
    let quantize = T::QUANTIZE.trim_end_matches('\0');
    assert_eq!(quantize, format!("launch_mmvq_gguf_quantize_q8_1_{act}"));
    assert!(source.contains(&format!("{quantize}(")), "{quantize}");
    for dtype in DTYPES {
        let name = T::matvec(dtype).expect("every dtype has a launcher");
        let tag = name
            .strip_prefix("launch_mmvq_gguf_")
            .and_then(|n| n.strip_suffix(&format!("_{act}_plain\0")))
            .unwrap_or_else(|| panic!("{name:?} is not a {act} launcher name"));
        let launcher = format!("MMVQ_LAUNCHER_PLAIN({tag}, {act},");
        assert!(source.contains(&launcher), "{launcher} not in mmvq_gguf.cu");
    }
}

/// Quantize random `(n, k)` weights and build an `(m, k)` activation in `act`.
fn setup(
    device: &Device,
    dtype: GgmlDType,
    (m, k, n): (usize, usize, usize),
    act: DType,
) -> Result<(QTensor, Tensor)> {
    let lhs: Vec<f32> = (0..m * k).map(|i| (i as f32 / 53.).sin()).collect();
    let rhs: Vec<f32> = (0..n * k).map(|i| (i as f32 / 71.).cos()).collect();
    let lhs = Tensor::from_slice(&lhs, (m, k), device)?.to_dtype(act)?;
    // `QTensor` stores the weights transposed, i.e. `(n, k)`.
    let rhs = Tensor::from_slice(&rhs, (n, k), device)?;
    Ok((QTensor::quantize(&rhs, dtype)?, lhs))
}

/// Assert `got` matches the dequantized matmul of `lhs` with `qt`.
///
/// The tolerance is relative to the largest magnitude in the *whole*
/// reference, not each row's. The `q8_1` error follows the size of the terms
/// being summed, and a row with a single output (`n = 1`) that happens to
/// cancel to near zero would otherwise be held to a bound a fraction of that
/// error.
fn assert_matches_reference(qt: &QTensor, lhs: &Tensor, got: &Tensor, what: &str) -> Result<()> {
    let want = lhs
        .to_dtype(DType::F32)?
        .matmul(&qt.dequantize(lhs.device())?.t()?)?
        .to_vec2::<f32>()?;
    let got = got.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    assert_eq!(got.len(), want.len(), "{what}");
    let scale = want
        .iter()
        .flatten()
        .fold(1.0f32, |acc, v| acc.max(v.abs()));
    for (row, (got, want)) in got.iter().zip(&want).enumerate() {
        assert_eq!(got.len(), want.len(), "{what}");
        for (col, (a, b)) in got.iter().zip(want).enumerate() {
            assert!(
                (a - b).abs() <= 2e-2 * scale,
                "{what} at ({row}, {col}): mmvq_gguf {a} vs dequantized {b} (row scale {scale})"
            );
        }
    }
    Ok(())
}

/// Run [`fast_mmvq::try_fwd`] on `qt` and `lhs`, returning the result as a
/// tensor in the activation dtype, or `None` if the path declined.
fn try_fwd(qt: &QTensor, lhs: &Tensor) -> Result<Option<Tensor>> {
    let QStorage::Rocm(q) = &qt.storage else {
        return Ok(None);
    };
    let (n, k) = qt.shape().dims2()?;
    let (m, _) = lhs.dims2()?;
    let out = match &*lhs.storage() {
        Storage::Rocm(s) => fast_mmvq::try_fwd(q, n, k, m, s, lhs.layout())?,
        _ => return Ok(None),
    };
    Ok(out.map(|s| {
        crate::tensor::from_storage(
            Storage::Rocm(s),
            (m, n),
            crate::op::BackpropOp::none(),
            false,
        )
    }))
}

/// The kernels themselves, for every dtype, batch and activation dtype: this
/// calls `mul_mat_vec` directly, so the DMMV gate in `try_fwd` does not
/// decide which cases run. `k = 4096` runs more than one iteration of the
/// kernel's column loop and `n = 128` spans several blocks.
#[test]
fn every_dtype_and_batch_matches_a_dequantized_matmul() -> Result<()> {
    let device = rocm_device!();
    let Device::Rocm(dev) = &device else {
        return Ok(());
    };
    for dtype in DTYPES {
        for (k, n) in [(1024usize, 64usize), (4096, 128)] {
            for m in [1usize, 2, 3, 4, 5, 8] {
                let (qt, lhs32) = setup(&device, dtype, (m, k, n), DType::F32)?;
                let QStorage::Rocm(q) = &qt.storage else {
                    return Ok(());
                };
                let flat = lhs32.flatten_all()?.to_vec1::<f32>()?;
                let what = format!("{dtype:?} m={m} k={k} n={n}");

                let y = dev.clone_htod(&flat)?;
                let dst = fast_mmvq::mul_mat_vec::<f32>(q, &y, 0, (k, n), m)?;
                let got = Tensor::from_vec(dev.clone_dtoh(&dst)?, (m, n), &Device::Cpu)?;
                assert_matches_reference(&qt, &lhs32, &got.to_device(&device)?, &what)?;

                let half: Vec<half::f16> = flat.iter().map(|&v| half::f16::from_f32(v)).collect();
                let y = dev.clone_htod(&half)?;
                let dst = fast_mmvq::mul_mat_vec::<half::f16>(q, &y, 0, (k, n), m)?;
                let got = Tensor::from_vec(dev.clone_dtoh(&dst)?, (m, n), &Device::Cpu)?;
                assert_matches_reference(
                    &qt,
                    &lhs32,
                    &got.to_device(&device)?,
                    &format!("{what} f16"),
                )?;

                let brain: Vec<half::bf16> =
                    flat.iter().map(|&v| half::bf16::from_f32(v)).collect();
                let y = dev.clone_htod(&brain)?;
                let dst = fast_mmvq::mul_mat_vec::<half::bf16>(q, &y, 0, (k, n), m)?;
                let got = Tensor::from_vec(dev.clone_dtoh(&dst)?, (m, n), &Device::Cpu)?;
                assert_matches_reference(
                    &qt,
                    &lhs32,
                    &got.to_device(&device)?,
                    &format!("{what} bf16"),
                )?;
            }
        }
    }
    Ok(())
}

/// The result keeps the activation's dtype, which is the point of the path.
#[test]
fn try_fwd_keeps_the_activation_dtype() -> Result<()> {
    let device = rocm_device!();
    for (act, ms) in [
        (DType::F16, [1usize, 4, 8]),
        (DType::BF16, [1, 4, 8]),
        (DType::F32, [2, 4, 8]),
    ] {
        let (qt, _) = setup(&device, GgmlDType::Q4_0, (ms[0], 1024, 64), act)?;
        for m in ms {
            let (_, lhs_m) = setup(&device, GgmlDType::Q4_0, (m, 1024, 64), act)?;
            let out = try_fwd(&qt, &lhs_m)?.expect("fast path should cover this");
            assert_eq!(out.dtype(), act, "m={m}");
            assert_matches_reference(&qt, &lhs_m, &out, &format!("{act:?} m={m}"))?;
        }
    }
    Ok(())
}

/// An odd `n` at a batch of two or more leaves the last block a row that does
/// not exist. The kernel drops that row's store; one that got through would
/// overwrite the next batch row's first output, which this catches. `n = 4097`
/// puts the last block far enough past one resident grid for such a store to
/// survive.
///
/// It cannot catch the phantom row's *load* running past the weights. That
/// read is discarded and the allocation's neighbours are normally mapped, so
/// the output is right either way; only a fault, or a run under a device
/// memory checker, shows it. `k = 4096` makes the unclamped read a whole
/// weight row, well past the 512 elements of weight padding, to give such a
/// checker something to see.
#[test]
fn odd_output_rows_stay_correct() -> Result<()> {
    let device = rocm_device!();
    for dtype in [GgmlDType::Q4_0, GgmlDType::Q4K] {
        for n in [1usize, 3, 65, 4097] {
            for m in [2usize, 5, 8] {
                let (qt, lhs) = setup(&device, dtype, (m, 4096, n), DType::F16)?;
                let out = try_fwd(&qt, &lhs)?.expect("fast path should cover this");
                assert_matches_reference(&qt, &lhs, &out, &format!("{dtype:?} m={m} n={n}"))?;
            }
        }
    }
    Ok(())
}

/// The `(b, m, k)` spelling flattens to the same batch as `(b * m, k)`, and a
/// 3-D activation is what a transformer actually hands the matmul.
#[test]
fn three_dimensional_activations_match() -> Result<()> {
    let device = rocm_device!();
    let (b, m, k, n) = (2usize, 3usize, 512usize, 64usize);
    for dtype in [GgmlDType::Q4_0, GgmlDType::Q4K] {
        let (qt, lhs) = setup(&device, dtype, (b * m, k, n), DType::F32)?;
        let lhs = lhs.reshape((b, m, k))?;
        let got = QMatMul::from_qtensor(qt)?.forward(&lhs)?;
        assert_eq!(got.dims(), &[b, m, n], "{dtype:?}");

        // `setup` is deterministic, so a second call is the same weights.
        let (reference, lhs_2d) = setup(&device, dtype, (b * m, k, n), DType::F32)?;
        assert_matches_reference(
            &reference,
            &lhs_2d,
            &got.reshape((b * m, n))?,
            &format!("{dtype:?} 3-D"),
        )?;
    }
    Ok(())
}

/// An activation `narrow`ed out of a larger buffer starts at a non-zero
/// offset, which has to scale by the element size.
#[test]
fn an_offset_activation_is_read_from_its_offset() -> Result<()> {
    let device = rocm_device!();
    let (qt, lhs) = setup(&device, GgmlDType::Q4K, (6, 1024, 64), DType::BF16)?;
    let view = lhs.narrow(0, 2, 3)?;
    assert_ne!(view.layout().start_offset(), 0);
    let out = try_fwd(&qt, &view)?.expect("fast path should cover this");
    assert_matches_reference(&qt, &view, &out, "offset view")
}

/// What the path leaves to the others: past the batch table, a batch of one
/// f32 row DMMV serves, and a view with gaps in it.
#[test]
fn declines_what_it_does_not_cover() -> Result<()> {
    let device = rocm_device!();
    let (qt, lhs) = setup(&device, GgmlDType::Q4_0, (9, 512, 32), DType::F32)?;
    assert!(try_fwd(&qt, &lhs)?.is_none(), "batch 9");

    let (qt, lhs) = setup(&device, GgmlDType::Q4K, (1, 512, 32), DType::F32)?;
    assert!(try_fwd(&qt, &lhs)?.is_none(), "f32 batch 1, K-quant");

    let (qt, lhs) = setup(&device, GgmlDType::Q4_0, (4, 1024, 32), DType::F32)?;
    let strided = lhs.narrow(1, 0, 512)?;
    assert!(try_fwd(&qt, &strided)?.is_none(), "non-contiguous");
    Ok(())
}

/// `QMatMul` end to end: the output of a bf16 matmul is bf16.
#[test]
fn qmatmul_returns_the_activation_dtype() -> Result<()> {
    let device = rocm_device!();
    let (qt, lhs) = setup(&device, GgmlDType::Q6K, (4, 1024, 64), DType::BF16)?;
    // `setup` is deterministic, so a second call is the same weights.
    let (reference, _) = setup(&device, GgmlDType::Q6K, (4, 1024, 64), DType::BF16)?;
    let out = QMatMul::from_qtensor(qt)?.forward(&lhs)?;
    assert_eq!(out.dtype(), DType::BF16);
    assert_matches_reference(&reference, &lhs, &out, "Q6K bf16 via QMatMul")
}
