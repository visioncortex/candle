//! Per-op timings of a convolutional net's building blocks on one device,
//! for comparing backends: `cargo run --release -p candle-core --features
//! vulkan --example conv_bench -- vulkan` (or `rocm`, `cuda`, `metal`, `cpu`).
//!
//! The shapes are the stages of a shallow U-Net at a 512-px tile with widths
//! 64/128/256 (the vtracer upscaler); pass other `C_IN,C_OUT,H,W` quadruples
//! after the device name to time your own. Every op is synced before and
//! after timing, so the figures are wall time per call, kernel plus launch.
use candle_core::{Device, Result, Tensor};
use std::time::Instant;

fn device(name: &str) -> Result<Device> {
    match name {
        "cpu" => Ok(Device::Cpu),
        "cuda" => Device::new_cuda(0),
        "metal" => Device::new_metal(0),
        #[cfg(feature = "rocm")]
        "rocm" => Device::new_rocm(0),
        "vulkan" => Device::new_vulkan(0),
        other => candle_core::bail!("unknown device {other}"),
    }
}

fn sync(dev: &Device) -> Result<()> {
    dev.synchronize()
}

/// Time `f` (which returns a tensor so nothing is optimised away): two
/// warm-up calls, then the mean of `iters`, with the device synced.
fn time<F: FnMut() -> Result<Tensor>>(dev: &Device, iters: usize, mut f: F) -> Result<f64> {
    for _ in 0..2 {
        let t = f()?;
        drop(t);
    }
    sync(dev)?;
    let t0 = Instant::now();
    for _ in 0..iters {
        let t = f()?;
        drop(t);
    }
    sync(dev)?;
    Ok(t0.elapsed().as_secs_f64() * 1e3 / iters as f64)
}

#[cfg(feature = "vulkan")]
fn stats_line() -> String {
    let s = candle_core::vulkan_backend::stats();
    candle_core::vulkan_backend::reset_stats();
    format!(
        "  [{} dispatches, {} syncs, gpu wait {:.1} ms, {} fallbacks]",
        s.native_dispatches,
        s.syncs,
        s.gpu_wait_ns as f64 / 1e6,
        s.cpu_fallbacks
    )
}
#[cfg(not(feature = "vulkan"))]
fn stats_line() -> String {
    String::new()
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let name = args.first().map(String::as_str).unwrap_or("cpu");
    let dev = device(name)?;
    println!("device: {dev:?}");
    let iters = 10;

    // (c_in, c_out, h, w): the 3x3 convs of the 512-tile U-Net, which works
    // on the pixel-unshuffled input (head 24->64 at 256^2, then 64@256^2,
    // 128@128^2, 256@64^2, tail 64->20 at 256^2).
    let mut shapes: Vec<(usize, usize, usize, usize)> = vec![
        (24, 64, 256, 256),
        (64, 64, 256, 256),
        (128, 128, 128, 128),
        (256, 256, 64, 64),
        (64, 20, 256, 256),
    ];
    for a in &args[1.min(args.len())..] {
        let v: Vec<usize> = a.split(',').filter_map(|s| s.parse().ok()).collect();
        if v.len() == 4 {
            shapes.push((v[0], v[1], v[2], v[3]));
        }
    }

    println!("{:<34} {:>10} {:>10}", "op", "ms/call", "GFLOP/s");
    for &(ci, co, h, w) in &shapes {
        let x = Tensor::randn(0f32, 1., (1, ci, h, w), &dev)?;
        let k = Tensor::randn(0f32, 0.1, (co, ci, 3, 3), &dev)?;
        let ms = time(&dev, iters, || x.conv2d(&k, 1, 1, 1, 1))?;
        let flop = 2.0 * (ci * co * 9 * h * w) as f64;
        println!(
            "{:<34} {:>10.2} {:>10.0}{}",
            format!("conv3x3 {ci}->{co} @{h}x{w}"),
            ms,
            flop / ms / 1e6,
            stats_line()
        );
    }
    // 1x1 convs: the pixel-unshuffle downs and pixel-shuffle ups.
    for &(ci, co, h, w) in &[
        (256usize, 128usize, 128usize, 128usize),
        (512, 256, 64, 64),
        (64, 128, 128, 128),
        (32, 64, 256, 256),
    ] {
        let x = Tensor::randn(0f32, 1., (1, ci, h, w), &dev)?;
        let k = Tensor::randn(0f32, 0.1, (co, ci, 1, 1), &dev)?;
        let ms = time(&dev, iters, || x.conv2d(&k, 0, 1, 1, 1))?;
        let flop = 2.0 * (ci * co * h * w) as f64;
        println!(
            "{:<34} {:>10.2} {:>10.0}{}",
            format!("conv1x1 {ci}->{co} @{h}x{w}"),
            ms,
            flop / ms / 1e6,
            stats_line()
        );
    }
    // Pooling, upsampling, elementwise, group norm.
    let x = Tensor::randn(0f32, 1., (1, 64, 256, 256), &dev)?;
    let ms = time(&dev, iters, || x.avg_pool2d((2, 2)))?;
    println!(
        "{:<34} {:>10.2} {:>10}{}",
        "avg_pool2d 2x2 64@256",
        ms,
        "",
        stats_line()
    );
    let y = Tensor::randn(0f32, 1., (1, 3, 256, 256), &dev)?;
    let ms = time(&dev, iters, || y.upsample_nearest2d(512, 512))?;
    println!(
        "{:<34} {:>10.2} {:>10}{}",
        "upsample_nearest2d 2x 3@256",
        ms,
        "",
        stats_line()
    );
    let ms = time(&dev, iters, || x.relu())?;
    println!(
        "{:<34} {:>10.2} {:>10}{}",
        "relu 64@256",
        ms,
        "",
        stats_line()
    );
    let x2 = Tensor::randn(0f32, 1., (1, 64, 256, 256), &dev)?;
    let ms = time(&dev, iters, || &x + &x2)?;
    println!(
        "{:<34} {:>10.2} {:>10}{}",
        "add 64@256",
        ms,
        "",
        stats_line()
    );
    // group norm's reductions: 4 groups of 16 channels.
    let ms = time(&dev, iters, || {
        x.reshape((1, 4, 16 * 256 * 256))?
            .sum_keepdim(2)?
            .broadcast_as((1, 4, 16 * 256 * 256))?
            .contiguous()
    })?;
    println!(
        "{:<34} {:>10.2} {:>10}{}",
        "sum_keepdim+broadcast 64@256",
        ms,
        "",
        stats_line()
    );
    // pixel shuffle: (1, 64, 128, 128) -> (1, 16, 256, 256)
    let z = Tensor::randn(0f32, 1., (1, 64, 128, 128), &dev)?;
    let ms = time(&dev, iters, || {
        z.reshape((1, 16, 2, 2, 128, 128))?
            .permute((0, 1, 4, 2, 5, 3))?
            .contiguous()?
            .reshape((1, 16, 256, 256))
    })?;
    println!(
        "{:<34} {:>10.2} {:>10}{}",
        "pixel_shuffle 64@128 -> 16@256",
        ms,
        "",
        stats_line()
    );
    Ok(())
}
