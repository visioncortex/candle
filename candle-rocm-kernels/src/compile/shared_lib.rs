//! Compiling and loading the shared kernel sources whose launchers live on
//! the host: the AMD WMMA MMQ kernels in `candle-kernels/src/mmq_gguf` and
//! the decode mat-vec kernels in `candle-kernels/src/mmvq_gguf.cu`.
//!
//! Unlike this crate's other kernels, these sources compile to a shared
//! library rather than a `--genco` device-only code object: see
//! [`hipcc::compile_shared_library`] for why. The result is opened with
//! `libloading` and its `extern "C"` launcher/quantize symbols are resolved
//! by name per call, the same "resolve fresh, cache only the loaded unit"
//! shape [`super::KernelCache::function`] already uses for
//! `hipModuleGetFunction`.

use super::{cache, detect, hipcc};
use crate::error::KernelError;
use libloading::Library;
use std::fs;
use std::path::Path;

/// One shared-library translation unit this crate knows how to compile.
///
/// Each variant is its own shared library: `mmq_gguf`'s quantize kernel and
/// per-dtype instance files were already separate `.cu` files upstream, and
/// nothing is gained by combining them into one compile.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum SharedLibUnit {
    /// `mmq_quantize.cu`. `f32 -> block_q8_1_mmq` activation requantization,
    /// shared by every quant type below.
    Quantize,
    /// `mmq_instance_q4_0.cu`.
    Q4_0,
    /// `mmq_instance_q4_1.cu`.
    Q4_1,
    /// `mmq_instance_q5_0.cu`.
    Q5_0,
    /// `mmq_instance_q5_1.cu`.
    Q5_1,
    /// `mmq_instance_q8_0.cu`.
    Q8_0,
    /// `mmq_instance_q2_k.cu`.
    Q2K,
    /// `mmq_instance_q3_k.cu`.
    Q3K,
    /// `mmq_instance_q4_k.cu`.
    Q4K,
    /// `mmq_instance_q5_k.cu`.
    Q5K,
    /// `mmq_instance_q6_k.cu`.
    Q6K,
    /// `mmvq_gguf.cu`. The batch 1-8 mat-vec kernels for every quant type
    /// above, and the f32/f16/bf16 -> `block_q8_1` quantizers they read.
    ///
    /// Its launchers return `void` and report nothing. A `b_size` outside
    /// 1..=8 launches no kernel and leaves `dst` untouched, and a failed
    /// launch is not surfaced. Callers must bound `b_size` themselves and
    /// check `hipGetLastError` after the call, as `quantized/rocm/mmq_wmma.rs`
    /// does for the MMQ units.
    Mmvq,
}

impl SharedLibUnit {
    /// `(source, name)` for this unit, the only place the two are tied
    /// together: adding a unit means adding one arm here, one in
    /// [`Self::is_mmq`] (both exhaustive, so the compiler catches a missed
    /// one) and its source constant in `lib.rs`.
    fn source_and_name(self) -> (&'static str, &'static str) {
        match self {
            SharedLibUnit::Quantize => (crate::MMQ_GGUF_QUANTIZE_SRC, "mmq_gguf_quantize"),
            SharedLibUnit::Q4_0 => (crate::MMQ_GGUF_Q4_0_SRC, "mmq_gguf_q4_0"),
            SharedLibUnit::Q4_1 => (crate::MMQ_GGUF_Q4_1_SRC, "mmq_gguf_q4_1"),
            SharedLibUnit::Q5_0 => (crate::MMQ_GGUF_Q5_0_SRC, "mmq_gguf_q5_0"),
            SharedLibUnit::Q5_1 => (crate::MMQ_GGUF_Q5_1_SRC, "mmq_gguf_q5_1"),
            SharedLibUnit::Q8_0 => (crate::MMQ_GGUF_Q8_0_SRC, "mmq_gguf_q8_0"),
            SharedLibUnit::Q2K => (crate::MMQ_GGUF_Q2_K_SRC, "mmq_gguf_q2_k"),
            SharedLibUnit::Q3K => (crate::MMQ_GGUF_Q3_K_SRC, "mmq_gguf_q3_k"),
            SharedLibUnit::Q4K => (crate::MMQ_GGUF_Q4_K_SRC, "mmq_gguf_q4_k"),
            SharedLibUnit::Q5K => (crate::MMQ_GGUF_Q5_K_SRC, "mmq_gguf_q5_k"),
            SharedLibUnit::Q6K => (crate::MMQ_GGUF_Q6_K_SRC, "mmq_gguf_q6_k"),
            SharedLibUnit::Mmvq => (crate::MMVQ_GGUF_SRC, "mmvq_gguf"),
        }
    }

    /// Whether this unit is one of `mmq_gguf`'s, which need that directory's
    /// headers and the WMMA define. `mmvq_gguf.cu` includes neither, so
    /// staging them would only tie its cache key to files it never reads.
    // Exhaustive on purpose: a new unit must be classified here, not default
    // to one family.
    #[allow(clippy::match_like_matches_macro)]
    fn is_mmq(self) -> bool {
        match self {
            SharedLibUnit::Quantize
            | SharedLibUnit::Q4_0
            | SharedLibUnit::Q4_1
            | SharedLibUnit::Q5_0
            | SharedLibUnit::Q5_1
            | SharedLibUnit::Q8_0
            | SharedLibUnit::Q2K
            | SharedLibUnit::Q3K
            | SharedLibUnit::Q4K
            | SharedLibUnit::Q5K
            | SharedLibUnit::Q6K => true,
            SharedLibUnit::Mmvq => false,
        }
    }

    fn source(self) -> &'static str {
        self.source_and_name().0
    }

    pub(crate) fn name(self) -> &'static str {
        self.source_and_name().1
    }
}

/// A loaded [`SharedLibUnit`].
///
/// The field is private and [`compile_and_load`] is the only constructor,
/// the same shape [`crate::wrappers::SendSyncModule`] uses for the same
/// reason: the `unsafe impl`s below assert a property of *this* wrapper, not
/// of `libloading::Library` in general.
pub struct SharedLibrary {
    lib: Library,
    /// The unit's name, kept only to say which library a failed lookup hit.
    name: &'static str,
}

// SAFETY: this wrapper's only state is a `dlopen` handle (plus a `'static`
// name), process-wide and not tied to the thread that opened it. `dlsym` is
// safe to call from any thread against a handle another thread opened, and
// `Library::get`'s returned `Symbol` borrows `&self` rather than carrying
// interior mutability, so concurrent lookups through a shared
// `&SharedLibrary` cannot race.
unsafe impl Send for SharedLibrary {}
// SAFETY: see the `Send` impl above.
unsafe impl Sync for SharedLibrary {}

impl SharedLibrary {
    /// Resolve `symbol` (NUL-terminated, e.g. `b"launch_mmq_gguf_q4_k\0"`) to
    /// a typed function pointer.
    ///
    /// Not cached. Like [`super::KernelCache::function`], a `dlsym` lookup is
    /// cheap next to the kernel launch it is about to drive, and caching the
    /// resolved pointer would need the same unsafe lifetime extension
    /// `KernelCache` deliberately avoids there.
    ///
    /// # Safety
    /// `T` must match the C function's real signature exactly: argument
    /// types, argument count and calling convention. A mismatch is undefined
    /// behavior that will not be caught at the call site.
    pub unsafe fn get<T>(&self, symbol: &[u8]) -> Result<libloading::Symbol<'_, T>, KernelError> {
        self.lib.get(symbol).map_err(|e| {
            KernelError::Compilation(format!(
                "symbol `{}` not found in shared library `{}`: {e}",
                String::from_utf8_lossy(symbol),
                self.name
            ))
        })
    }
}

/// Headers staged for `unit`'s compile: the `hip_shim` entries every kernel
/// needs (`mmq_common.cuh` and `mmvq_gguf.cu` both `#include`
/// `cuda_fp16.h`/`cuda_bf16.h`, which the shim shadows), plus `mmq_gguf`'s
/// own four headers for an MMQ unit.
fn staged_headers(unit: SharedLibUnit) -> Vec<(&'static str, &'static str)> {
    let mmq: &[(&str, &str)] = if unit.is_mmq() {
        crate::MMQ_GGUF_HEADERS
    } else {
        &[]
    };
    crate::HEADERS
        .iter()
        .filter(|(name, _)| name.starts_with("hip_shim/"))
        .chain(mmq)
        .copied()
        .collect()
}

/// The hipcc flags `unit`'s compile needs beyond `-O3 -std=c++17`.
///
/// `GGML_USE_HIP` goes on unconditionally: every `defined(GGML_USE_HIP)`
/// branch in these sources is how they tell "compiling for AMD" apart from
/// "compiling for NVIDIA", matching upstream llama.cpp. `__CUDA_ARCH__` is set
/// to the same value as [`cache::COMPILE_FLAGS`], for the same reason: several
/// `__CUDA_ARCH__`-gated branches in `mmq_common.cuh` (`ggml_cuda_dp4a`,
/// `warp_reduce_sum`/`warp_reduce_max`, `ggml_cuda_get_max_cpy_bytes`) are not
/// behind a `GGML_USE_HIP` guard and fall back to a slower path if left
/// undefined; the intrinsics they gate towards (`__dp4a`, `__reduce_add_sync`,
/// `__reduce_max_sync`) are all shimmed in `hip_compat.h`; `mmvq_gguf.cu`'s own
/// `ggml_cuda_dp4a` has the same gate. The `RDNA*`
/// architecture define follows [`detect::rdna_define`], the same one
/// `quantized.cu` uses. `AMD_WMMA_AVAILABLE` is added only for the
/// MMQ units on the architectures [`detect::has_wmma`] reports, which gates
/// the actual matrix-core device code in `mmq_mma.cuh` and the host-side tile
/// sizing in `mmq_common.cuh`'s `amd_wmma_available`.
fn compile_flags(arch: &str, unit: SharedLibUnit) -> Vec<String> {
    let mut flags = vec![
        "-DGGML_USE_HIP".to_string(),
        "-D__CUDA_ARCH__=890".to_string(),
    ];
    if let Some(define) = detect::rdna_define(arch) {
        flags.push(format!("-D{define}"));
    }
    if unit.is_mmq() && detect::has_wmma(arch) {
        flags.push("-DAMD_WMMA_AVAILABLE".to_string());
    }
    flags
}

/// Compile (or read from the disk cache) `unit`'s shared library for `arch`,
/// then `dlopen` it.
pub(crate) fn compile_and_load(
    cache_dir: &Path,
    arch: &str,
    toolchain: &str,
    unit: SharedLibUnit,
) -> Result<SharedLibrary, KernelError> {
    let headers = staged_headers(unit);
    let src_dir = cache_dir.join(format!("shared-lib-src-{}", cache::headers_key(&headers)));
    hipcc::stage_headers(&src_dir, &headers)?;

    let flags = compile_flags(arch, unit);
    let flag_refs: Vec<&str> = flags.iter().map(String::as_str).collect();

    let key = cache::shared_lib_key(unit.source(), &flag_refs, &headers, arch, toolchain);
    let entry = format!("{}_{key}", unit.name());
    let so_path = cache_dir.join(format!("{entry}.so"));
    let forced = cache::force_recompile();

    if forced || fs::metadata(&so_path).is_err() {
        let _lock = cache::lock_entry(cache_dir, &entry)?;
        if forced || fs::metadata(&so_path).is_err() {
            let tmp_so = cache_dir.join(format!("{entry}.{}.so.tmp", std::process::id()));
            hipcc::compile_shared_library(
                &src_dir,
                &tmp_so,
                arch,
                unit.name(),
                &entry,
                unit.source(),
                &flag_refs,
            )?;
            fs::rename(&tmp_so, &so_path).map_err(|e| {
                let _ = fs::remove_file(&tmp_so);
                KernelError::Io(format!("could not place {}: {e}", so_path.display()))
            })?;
        }
    }

    // SAFETY: `so_path` is a shared library this process just built (or
    // previously built and verified it could build) from candle-rocm's own
    // vendored, trusted source, not an arbitrary path. `dlopen` runs the
    // library's static initializers, which is the inherent risk any
    // `Library::new` call carries; there is nothing library-specific beyond
    // that here.
    unsafe { Library::new(&so_path) }
        .map(|lib| SharedLibrary {
            lib,
            name: unit.name(),
        })
        .map_err(|e| {
            KernelError::Compilation(format!("could not dlopen {}: {e}", so_path.display()))
        })
}

#[cfg(test)]
mod tests {
    use super::{compile_flags, staged_headers, SharedLibUnit};

    /// `mmvq_gguf.cu` gets the shim and nothing from `mmq_gguf/`, while an
    /// MMQ unit still gets both, in the order its cache key was built from.
    #[test]
    fn only_mmq_units_stage_the_mmq_headers() {
        let mmvq = staged_headers(SharedLibUnit::Mmvq);
        assert!(mmvq.iter().all(|(name, _)| name.starts_with("hip_shim/")));
        assert!(mmvq
            .iter()
            .any(|(name, _)| *name == "hip_shim/hip_compat.h"));

        let q4k = staged_headers(SharedLibUnit::Q4K);
        let (shim, rest) = q4k.split_at(mmvq.len());
        assert_eq!(shim, &mmvq[..]);
        assert_eq!(rest, crate::MMQ_GGUF_HEADERS);
    }

    /// The WMMA define is for `mmq_mma.cuh`; `mmvq_gguf.cu` must not get it,
    /// but it does need `__CUDA_ARCH__` for the `__dp4a` path.
    #[test]
    fn only_mmq_units_get_the_wmma_define() {
        let mmvq = compile_flags("gfx1101", SharedLibUnit::Mmvq);
        assert!(!mmvq.iter().any(|f| f == "-DAMD_WMMA_AVAILABLE"));
        assert!(mmvq.iter().any(|f| f == "-D__CUDA_ARCH__=890"));

        let q4k = compile_flags("gfx1101", SharedLibUnit::Q4K);
        assert!(q4k.iter().any(|f| f == "-DAMD_WMMA_AVAILABLE"));
    }
}
