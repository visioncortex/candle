//! Lists every Vulkan physical device's memory types and heaps, the raw
//! facts behind the backend's memory-type choice:
//! `cargo run --release -p candle-core --features vulkan --example vk_memtypes`.
use ash::vk;

fn main() {
    let entry = unsafe { ash::Entry::load() }.expect("vulkan loader");
    let info = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_2);
    let instance = unsafe {
        entry.create_instance(&vk::InstanceCreateInfo::default().application_info(&info), None)
    }
    .expect("instance");
    for (i, p) in unsafe { instance.enumerate_physical_devices() }
        .expect("devices")
        .iter()
        .enumerate()
    {
        let props = unsafe { instance.get_physical_device_properties(*p) };
        let name = props.device_name_as_c_str().map(|s| s.to_string_lossy().into_owned());
        println!(
            "[{i}] {:?} type {}",
            name.unwrap_or_default(),
            props.device_type.as_raw()
        );
        let mem = unsafe { instance.get_physical_device_memory_properties(*p) };
        for (h, heap) in mem.memory_heaps[..mem.memory_heap_count as usize].iter().enumerate() {
            println!(
                "    heap {h}: {:>7} MB flags {:#x}",
                heap.size >> 20,
                heap.flags.as_raw()
            );
        }
        for (t, ty) in mem.memory_types[..mem.memory_type_count as usize].iter().enumerate() {
            let f = ty.property_flags;
            let mut names = Vec::new();
            for (flag, n) in [
                (vk::MemoryPropertyFlags::DEVICE_LOCAL, "DEVICE_LOCAL"),
                (vk::MemoryPropertyFlags::HOST_VISIBLE, "HOST_VISIBLE"),
                (vk::MemoryPropertyFlags::HOST_COHERENT, "HOST_COHERENT"),
                (vk::MemoryPropertyFlags::HOST_CACHED, "HOST_CACHED"),
                (vk::MemoryPropertyFlags::LAZILY_ALLOCATED, "LAZILY_ALLOCATED"),
            ] {
                if f.contains(flag) {
                    names.push(n);
                }
            }
            println!("    type {t:>2}: heap {} {}", ty.heap_index, names.join("|"));
        }
    }
    unsafe { instance.destroy_instance(None) };
}
