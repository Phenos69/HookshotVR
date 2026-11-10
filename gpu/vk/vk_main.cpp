// The Vulkan back end's start: vulkan-1.dll, the instance, the device (Vulkan 1.3 and the extensions the NVN replay
// wants where the GPU has them: mesh shaders, host memory import, graphics pipeline libraries), the window's
// swapchain, and the rx_* table (vk_api.h). The NVN replay itself is vk_replay.cpp.
#include "vk_paths.h"
#include "vk_loader.h"

#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "vk_api.h"
#include "vk_draw.h"
#include "vk_framegen.h"
#include "vk_internal.h"
#include "gpu_choice.h"
#include "vk_menu.h"
#include "vk_dlss.h"
#include "vk_fsr4.h"
#include "vk_reflex.h"
#include "vk_antilag.h"
#include "vk_progress.h"
#include "vk_vr.h"

namespace vkx {
#define VKX_DEFINE(f) PFN_##f f;
VKX_GLOBAL(VKX_DEFINE)
VKX_INSTANCE(VKX_DEFINE)
VKX_DEVICE(VKX_DEFINE)
VKX_DEVICE2(VKX_DEFINE)
VKX_DEVICE3(VKX_DEFINE)
#undef VKX_DEFINE

static PFN_vkGetInstanceProcAddr get_instance_proc;

bool load_global() {
#ifdef _WIN32   // (hookshot: the loader from System32 only, not from the current folder or PATH)
    HMODULE m = LoadLibraryExA("vulkan-1.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
#else
    HMODULE m = LoadLibraryA("vulkan-1.dll");   // (linux: libvulkan.so.1, host/posix/win32_posix.c)
#endif
    if (!m) return false;
    get_instance_proc = (PFN_vkGetInstanceProcAddr)(void*)GetProcAddress(m, "vkGetInstanceProcAddr");
    if (!get_instance_proc) return false;
#define VKX_LOAD(f) f = (PFN_##f)get_instance_proc(nullptr, #f);
    VKX_GLOBAL(VKX_LOAD)
#undef VKX_LOAD
    return vkCreateInstance != nullptr;
}
PFN_vkGetInstanceProcAddr instance_proc_addr() { return get_instance_proc; }
void load_instance(VkInstance instance) {
#define VKX_LOAD(f) f = (PFN_##f)get_instance_proc(instance, #f);
    VKX_INSTANCE(VKX_LOAD)
#undef VKX_LOAD
}
void load_device(VkDevice device) {
#define VKX_LOAD(f) f = (PFN_##f)vkGetDeviceProcAddr(device, #f);
    VKX_DEVICE(VKX_LOAD)
    VKX_DEVICE2(VKX_LOAD)
    VKX_DEVICE3(VKX_LOAD)
#undef VKX_LOAD
}
} // namespace vkx

namespace mkwvk {

std::vector<std::string> gpu_device_names;
const MkwVkHost* host;
int64_t prof_ns[PROF_COUNT];
Device dev;
PFN_vkGetInstanceProcAddr instance_proc() { return vkx::instance_proc_addr(); }

void logf(const char* fmt, ...) {
    char b[1024];
    std::va_list a;
    va_start(a, fmt);
    std::vsnprintf(b, sizeof b, fmt, a);
    va_end(a);
    if (host) host->log(b);
}

namespace {

bool has_extension(const std::vector<VkExtensionProperties>& all, const char* name) {
    for (const auto& e : all)
        if (!std::strcmp(e.extensionName, name)) return true;
    return false;
}

bool create_instance() {
    uint32_t version = VK_API_VERSION_1_0;
    if (vkEnumerateInstanceVersion) vkEnumerateInstanceVersion(&version);
    if (version < VK_API_VERSION_1_3) {
        logf("vk: the Vulkan loader is %u.%u: 1.3 is needed", VK_VERSION_MAJOR(version), VK_VERSION_MINOR(version));
        return false;
    }
#ifdef _WIN32
    const char* exts[] = {VK_KHR_SURFACE_EXTENSION_NAME, VK_KHR_WIN32_SURFACE_EXTENSION_NAME};
    const uint32_t n_exts = 2;
#else
    // (linux: every window system's surface the loader has, for whichever one SDL's window is on: X11 or Wayland)
    const char* exts[4] = {VK_KHR_SURFACE_EXTENSION_NAME};
    uint32_t n_exts = 1;
    {
        uint32_t n = 0;
        vkEnumerateInstanceExtensionProperties(nullptr, &n, nullptr);
        std::vector<VkExtensionProperties> all(n);
        vkEnumerateInstanceExtensionProperties(nullptr, &n, all.data());
        for (const char* want : {"VK_KHR_xlib_surface", "VK_KHR_xcb_surface", "VK_KHR_wayland_surface"})
            if (has_extension(all, want)) exts[n_exts++] = want;
    }
#endif
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "mkw-recomp";
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ci.pApplicationInfo = &app;
    std::vector<const char*> ext_list(exts, exts + n_exts);
    {   // (hookshot) what NVIDIA DLSS needs (vk_dlss.cpp; nothing in a build without it)
        uint32_t n = 0;
        vkEnumerateInstanceExtensionProperties(nullptr, &n, nullptr);
        std::vector<VkExtensionProperties> all(n);
        vkEnumerateInstanceExtensionProperties(nullptr, &n, all.data());
        dlss_instance_extensions(ext_list, all);
        // (hookshot) the HDR10 colour space for the Insert menu's HDR (the menu's settings are read after the instance is made, so
        // whenever the driver has it; the SDR choice in create_swapchain keeps to the sRGB surface)
        if (has_extension(all, VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME))
            ext_list.push_back(VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME), dev.swapchain_colorspace = true;
    }
    ci.enabledExtensionCount = (uint32_t)ext_list.size();
    ci.ppEnabledExtensionNames = ext_list.data();
    // RX_VK_VALIDATION=1: the Khronos validation layer (when installed)
    const char* layer = "VK_LAYER_KHRONOS_validation";
    if (getenv("RX_VK_VALIDATION")) ci.enabledLayerCount = 1, ci.ppEnabledLayerNames = &layer;
    // (VR) the headset's runtime adds the instance extensions it needs
    const VkResult made = vr::active() ? vr::create_instance(&ci, &dev.instance) : vkCreateInstance(&ci, nullptr, &dev.instance);
    if (made != VK_SUCCESS) {
        logf("vk: vkCreateInstance failed");
        return false;
    }
    load_instance(dev.instance);
    return true;
}

bool pick_device() {
    menu_init();
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(dev.instance, &n, nullptr);
    std::vector<VkPhysicalDevice> all(n);
    vkEnumeratePhysicalDevices(dev.instance, &n, all.data());

    std::vector<uint32_t> valid_gpus;
    gpu_device_names.clear();
    for (uint32_t i = 0; i < n; i++) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(all[i], &p);
        logf("vk: device %u: %s, Vulkan %u.%u, driver %#x, type %d", i, p.deviceName, VK_API_VERSION_MAJOR(p.apiVersion),
             VK_API_VERSION_MINOR(p.apiVersion), p.driverVersion, (int)p.deviceType);
        if (p.apiVersion >= VK_API_VERSION_1_3) {
            valid_gpus.push_back(i);
            gpu_device_names.push_back(p.deviceName);
        }
    }
    menu_set_gpu_count((int)valid_gpus.size());
    if (vr::active()) {   // (VR) the GPU the headset is attached to, not the menu's choice
        if (const VkPhysicalDevice pd = vr::physical_device(dev.instance)) {
            VkPhysicalDeviceProperties vp;
            vkGetPhysicalDeviceProperties(pd, &vp);
            if (vp.apiVersion >= VK_API_VERSION_1_3) {
                logf("vr: using the headset's GPU: %s", vp.deviceName);
                dev.physical = pd;
                dev.props = vp;
                vkGetPhysicalDeviceMemoryProperties(dev.physical, &dev.memory);
                return true;
            }
            logf("vr: the headset's GPU (%s) lacks Vulkan 1.3: VR cannot work on it", vp.deviceName);
        }
    }

    // (AMD laptops) With AMD graphics in the processor and an AMD card, Windows has two AMD drivers installed, one an
    // adapter, and each one's Vulkan driver lists every AMD GPU: the card shows up twice. It came first through the
    // processor graphics' driver, which the card's own update leaves as it was: on an RX 7700S that older driver
    // (amdvlk64.dll 32.0.11036.4002) crashed in its shader compiler (0xc0000409) on the game's compute pipelines while
    // the card's 32.0.32015.7029 built them. The same kind of GPU twice: the one with the newer driver wins.
    // RX_VK_DEVICE=<n> or opts.gpu_device: that device, as listed in the log
    int best = -1, best_score = -1;
    uint32_t best_driver = 0;
    int target_raw_idx = -1;
    if (const char* want = getenv("RX_VK_DEVICE")) {
        if (*want) target_raw_idx = atoi(want);
    } else if (const int k = gpu_choice::menu_device(opts.gpu_device, (int)valid_gpus.size()); k >= 0) {
        target_raw_idx = (int)valid_gpus[k];
    }

    for (uint32_t i : valid_gpus) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(all[i], &p);
        if (target_raw_idx >= 0) {
            if ((int)i == target_raw_idx) best = (int)i, best_score = 3;
            continue;
        }
        const int score = p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 2 :
                          p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 1 : 0;
        if (score > best_score || (score == best_score && p.driverVersion > best_driver))
            best = (int)i, best_score = score, best_driver = p.driverVersion;
    }
    if (best < 0 && target_raw_idx >= 0) {
        logf("vk: preferred device %d not available or lacks Vulkan 1.3, falling back to auto selection", target_raw_idx);
        for (uint32_t i : valid_gpus) {
            VkPhysicalDeviceProperties p;
            vkGetPhysicalDeviceProperties(all[i], &p);
            const int score = p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 2 :
                              p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 1 : 0;
            if (score > best_score || (score == best_score && p.driverVersion > best_driver))
                best = (int)i, best_score = score, best_driver = p.driverVersion;
        }
    }
    if (best < 0) {
        logf("vk: no Vulkan 1.3 GPU");
        return false;
    }
    dev.physical = all[best];
    vkGetPhysicalDeviceProperties(dev.physical, &dev.props);
    vkGetPhysicalDeviceMemoryProperties(dev.physical, &dev.memory);
    return true;
}

bool create_device() {
    uint32_t n = 0;
    vkEnumerateDeviceExtensionProperties(dev.physical, nullptr, &n, nullptr);
    std::vector<VkExtensionProperties> exts(n);
    vkEnumerateDeviceExtensionProperties(dev.physical, nullptr, &n, exts.data());
    std::vector<const char*> want = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
    auto optional = [&](const char* name, bool& flag) {
        flag = has_extension(exts, name);
        if (flag) want.push_back(name);
    };
    // (RX_VK_FORCE_MESH_FALLBACK=1, a test: mesh shaders not used where they are, so the no-mesh fallback the GTX 10xx and RX 5000
    // cards run (issue 640) can be tried on any GPU)
    if (getenv("RX_VK_FORCE_MESH_FALLBACK") && atoi(getenv("RX_VK_FORCE_MESH_FALLBACK")) != 0) dev.mesh_shader = false;
    else optional(VK_EXT_MESH_SHADER_EXTENSION_NAME, dev.mesh_shader);
    optional(VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME, dev.external_memory_host);
    optional(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME, dev.external_memory_fd);   // (guest memory as udmabufs: vk_memory.cpp import_dma_buf)
    if (dev.external_memory_fd) optional(VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME, dev.dma_buf_import);
    optional(VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME, dev.pipeline_library);
    if (dev.pipeline_library) optional(VK_EXT_GRAPHICS_PIPELINE_LIBRARY_EXTENSION_NAME, dev.graphics_pipeline_library);
    optional(VK_EXT_CUSTOM_BORDER_COLOR_EXTENSION_NAME, dev.custom_border_color);
    optional(VK_EXT_EXTENDED_DYNAMIC_STATE_3_EXTENSION_NAME, dev.dynamic_state3);
    optional(VK_EXT_DEPTH_CLIP_CONTROL_EXTENSION_NAME, dev.depth_clip_control);
    optional(VK_EXT_ROBUSTNESS_2_EXTENSION_NAME, dev.null_descriptor);
    optional(VK_EXT_INDEX_TYPE_UINT8_EXTENSION_NAME, dev.index_uint8);
    optional(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME, dev.push_descriptor);
    optional(VK_EXT_CONDITIONAL_RENDERING_EXTENSION_NAME, dev.conditional_rendering);
    optional(VK_EXT_PROVOKING_VERTEX_EXTENSION_NAME, dev.provoking_vertex);
    optional(VK_EXT_VERTEX_ATTRIBUTE_DIVISOR_EXTENSION_NAME, dev.vertex_divisor);
    optional(VK_EXT_SHADER_STENCIL_EXPORT_EXTENSION_NAME, dev.stencil_export);
    optional(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME, dev.memory_budget);
    optional(VK_EXT_IMAGE_2D_VIEW_OF_3D_EXTENSION_NAME, dev.view_2d_of_3d);
    // (hookshot) on by default: with a marker before each draw and dispatch the device has not been lost (five
    // runs); without them it was, every run, shortly after OoT's first gameplay frames (nvlddmkm event 153; the
    // extension alone, RX_VK_CHECKPOINTS=ext, did not keep it). The race they hide is not found yet.
    // RX_VK_CHECKPOINTS=0: off
    if (!getenv("RX_VK_CHECKPOINTS") || strcmp(getenv("RX_VK_CHECKPOINTS"), "0") != 0)
        optional(VK_NV_DEVICE_DIAGNOSTIC_CHECKPOINTS_EXTENSION_NAME, dev.checkpoints);
    // (asked for only: a marker written twice a draw, where the NV ones are not there to hide a race)
    if (const char* c = getenv("RX_VK_CHECKPOINTS"); !dev.checkpoints && c && strcmp(c, "0") != 0 && strcmp(c, "ext") != 0)
        optional(VK_AMD_BUFFER_MARKER_EXTENSION_NAME, dev.buffer_marker);
    optional(VK_EXT_DEVICE_FAULT_EXTENSION_NAME, dev.device_fault);
    optional(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME, dev.coop_matrix);
    optional(VK_KHR_SWAPCHAIN_MUTABLE_FORMAT_EXTENSION_NAME, dev.swap_mutable);   // (hookshot: the overlays, vk_imgui.cpp)
    dlss_device_extensions(want, exts);   // (hookshot: NVIDIA DLSS's, vk_dlss.cpp)
    bool fsr_descriptor_buffer = false;   // (hookshot, linux: only with a Vulkan FSR 4 library beside the game, vk_fsr4.cpp)
    if (fsr4_wants_descriptor_buffer()) optional(VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME, fsr_descriptor_buffer);
    reflex_device_extensions(want, exts);   // (hookshot: NVIDIA Reflex's, vk_reflex.cpp)
    antilag_device_extensions(want, exts);   // (hookshot: AMD Anti-Lag 2's, vk_antilag.cpp)

    // the queue: one family for graphics, compute and transfer
    vkGetPhysicalDeviceQueueFamilyProperties(dev.physical, &n, nullptr);
    std::vector<VkQueueFamilyProperties> families(n);
    vkGetPhysicalDeviceQueueFamilyProperties(dev.physical, &n, families.data());
    dev.queue_family = UINT32_MAX;
    for (uint32_t i = 0; i < n && dev.queue_family == UINT32_MAX; i++)
        if ((families[i].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) ==
            (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT))
            dev.queue_family = i;
    if (dev.queue_family == UINT32_MAX) return false;
    const float priority = 1.0f;
    VkDeviceQueueCreateInfo q{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    q.queueFamilyIndex = dev.queue_family;
    q.queueCount = 1;
    q.pQueuePriorities = &priority;
    // (hookshot) The presents on a queue of a compute family of their own, beside the replay's (RX_VK_PRESENT_QUEUE=1 / 0: on /
    // off; on by default with NVIDIA's driver on Windows, where it was measured). On the replay's queue vkQueuePresentKHR blocked
    // with dev.present_mu held, ~0.8 ms a present with frame interpolation at 4x and up to 8, and the game's main and presentation
    // threads waited there to submit. A second queue of the graphics family took that lock off the presents, but its presents
    // blocked longer (~2-4 ms: the graphics work of the two queues takes turns on the GPU) and the made frames were shown late
    // four times as often. A compute family's queue runs beside the graphics work: in Hyrule Field (a 14700K and an RTX 5070 Ti,
    // 4x, 175 Hz) the game went from ~52 to ~58 fps standing and ~46.5 to ~53 running, its frames over 22 ms from ~13% to ~1%
    // standing and over 40 ms from ~4% to ~0.2% running, the made frames as evenly shown standing. The swapchain's images are
    // shared by both families (vk_main.cpp create_swapchain), and a family that cannot present to the window leaves it off.
    uint32_t compute_family = UINT32_MAX;
    {
        bool want = false;
#ifdef _WIN32
        VkPhysicalDeviceDriverProperties dp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
        VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        p2.pNext = &dp;
        vkGetPhysicalDeviceProperties2(dev.physical, &p2);
        want = false;   // (v2.0.9: off by default with the write-back diff until issue 1161's device lost is placed; was:
        (void)dp;       //  NVIDIA's driver on Windows. RX_VK_PRESENT_QUEUE=1: on)
#endif
        if (const char* e = getenv("RX_VK_PRESENT_QUEUE")) want = atoi(e) != 0;
        for (uint32_t i = 0; want && i < n && compute_family == UINT32_MAX; i++)
            if ((families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && !(families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT))
                compute_family = i;
    }
    VkDeviceQueueCreateInfo qs[2] = {q, {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}};
    qs[1].queueFamilyIndex = compute_family;
    qs[1].queueCount = 1;
    qs[1].pQueuePriorities = &priority;

    // features: what the hardware has of what the replay uses (all of them on the GPUs MKW-class hardware maps to)
    VkPhysicalDeviceMeshShaderFeaturesEXT mesh{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};
    VkPhysicalDeviceRobustness2FeaturesEXT robust2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT};
    VkPhysicalDeviceIndexTypeUint8FeaturesEXT u8{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INDEX_TYPE_UINT8_FEATURES_EXT};
    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan11Features f11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    VkPhysicalDeviceFeatures2 f{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    f.pNext = &f11, f11.pNext = &f12, f12.pNext = &f13;
    VkPhysicalDeviceExtendedDynamicState3FeaturesEXT eds3{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_FEATURES_EXT};
    VkPhysicalDeviceCustomBorderColorFeaturesEXT border{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUSTOM_BORDER_COLOR_FEATURES_EXT};
    VkPhysicalDeviceDepthClipControlFeaturesEXT clip{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_CLIP_CONTROL_FEATURES_EXT};
    VkPhysicalDeviceConditionalRenderingFeaturesEXT cond{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CONDITIONAL_RENDERING_FEATURES_EXT};
    VkPhysicalDeviceProvokingVertexFeaturesEXT pv{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROVOKING_VERTEX_FEATURES_EXT};
    VkPhysicalDeviceGraphicsPipelineLibraryFeaturesEXT gpl{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GRAPHICS_PIPELINE_LIBRARY_FEATURES_EXT};
    void** tail = &f13.pNext;
    if (dev.mesh_shader) *tail = &mesh, tail = &mesh.pNext;
    if (dev.null_descriptor) *tail = &robust2, tail = &robust2.pNext;
    if (dev.index_uint8) *tail = &u8, tail = &u8.pNext;
    if (dev.dynamic_state3) *tail = &eds3, tail = &eds3.pNext;
    if (dev.custom_border_color) *tail = &border, tail = &border.pNext;
    if (dev.depth_clip_control) *tail = &clip, tail = &clip.pNext;
    if (dev.conditional_rendering) *tail = &cond, tail = &cond.pNext;
    if (dev.provoking_vertex) *tail = &pv, tail = &pv.pNext;
    VkPhysicalDeviceVertexAttributeDivisorFeaturesEXT vdiv{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_ATTRIBUTE_DIVISOR_FEATURES_EXT};
    if (dev.vertex_divisor) *tail = &vdiv, tail = &vdiv.pNext;
    if (dev.graphics_pipeline_library) *tail = &gpl, tail = &gpl.pNext;
    VkPhysicalDeviceFaultFeaturesEXT fault{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FAULT_FEATURES_EXT};
    if (dev.device_fault) *tail = &fault, tail = &fault.pNext;
    VkPhysicalDeviceImage2DViewOf3DFeaturesEXT v2d3d{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_2D_VIEW_OF_3D_FEATURES_EXT};
    if (dev.view_2d_of_3d) *tail = &v2d3d, tail = &v2d3d.pNext;
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR coop{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
    if (dev.coop_matrix) *tail = &coop, tail = &coop.pNext;
    VkPhysicalDeviceDescriptorBufferFeaturesEXT descriptor_buffer{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_FEATURES_EXT};
    if (fsr_descriptor_buffer) *tail = &descriptor_buffer, tail = &descriptor_buffer.pNext;
    reflex_device_features(tail);   // (present ids, when NVIDIA Reflex's extensions are there)
    antilag_device_features(tail);   // (AMD Anti-Lag 2's, when its extension is there)
    vkGetPhysicalDeviceFeatures2(dev.physical, &f);
    descriptor_buffer.descriptorBufferCaptureReplay = VK_FALSE;
    dev.view_2d_of_3d = dev.view_2d_of_3d && v2d3d.sampler2DViewOf3D;
    coop.cooperativeMatrixRobustBufferAccess = VK_FALSE;
    dev.coop_matrix = dev.coop_matrix && coop.cooperativeMatrix &&
                      f12.vulkanMemoryModel && f12.vulkanMemoryModelDeviceScope;
    // (bounds-checked buffer access: an out-of-range access dropped instead of faulting the device. On by default: the GPU
    // culling pass 53caac5f faulted the device while a save loaded, about one start in three, and the scene's lighting read garbage
    // for single frames (the black flashes); with it, neither in A/B runs at the same GPU time. Lichii's OoT branch keeps it on too.
    // The 'Robust buffers' row of hookshot_options.ini chooses (0 Auto: on, on every vendor; 1 Off; 2 On); RX_VK_ROBUST=0 or 1
    // over it. Off is null descriptors only: some NVIDIA cards ran slower with it on, issues 394, 418, 419)
    static const bool robust = []() -> bool {
        if (const char* e = getenv("RX_VK_ROBUST")) return atoi(e) != 0;
        char exe[MAX_PATH];
        const DWORD n = game_module_path(exe, MAX_PATH);   // (the game folder, not bin beside the launcher: vk_paths.h)
        std::string ini(exe, n);
        ini = ini.substr(0, ini.find_last_of("\\/") + 1) + "hookshot_options.ini";
        int v = 0;   // (Auto)
        if (FILE* f = fopen(ini.c_str(), "r")) {
            char line[80];
            while (fgets(line, sizeof line, f))
                if (!strncmp(line, "robust_buffers=", 15)) {
                    v = atoi(line + 15);
                    break;
                }
            fclose(f);   // (once: the PR closed it twice when the row said Auto)
        }
        return v != 1;   // (Auto and On: on)
    }();
    const bool robust_ok = robust && robust2.robustBufferAccess2 && f.features.robustBufferAccess;
    robust2.robustBufferAccess2 = robust_ok ? VK_TRUE : VK_FALSE;
    robust2.robustImageAccess2 = VK_FALSE;
    dev.null_descriptor = dev.null_descriptor && robust2.nullDescriptor;
    pv.transformFeedbackPreservesProvokingVertex = VK_FALSE;
    dev.provoking_vertex = dev.provoking_vertex && pv.provokingVertexLast;
    dev.conditional_rendering = dev.conditional_rendering && cond.conditionalRendering;
    dev.custom_border_color = dev.custom_border_color && border.customBorderColors && border.customBorderColorWithoutFormat;
    dev.depth_clip_control = dev.depth_clip_control && clip.depthClipControl;
    dev.graphics_pipeline_library = dev.graphics_pipeline_library && gpl.graphicsPipelineLibrary;
    if (dev.graphics_pipeline_library) {
        VkPhysicalDeviceGraphicsPipelineLibraryPropertiesEXT lp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GRAPHICS_PIPELINE_LIBRARY_PROPERTIES_EXT};
        VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        p2.pNext = &lp;
        vkGetPhysicalDeviceProperties2(dev.physical, &p2);
        dev.graphics_pipeline_library = lp.graphicsPipelineLibraryFastLinking != VK_FALSE;
    }
    dev.eds3_polygon = dev.dynamic_state3 && eds3.extendedDynamicState3PolygonMode;
    dev.eds3_clamp = dev.dynamic_state3 && eds3.extendedDynamicState3DepthClampEnable;
    dev.eds3_blend = dev.dynamic_state3 && eds3.extendedDynamicState3ColorBlendEnable &&
                     eds3.extendedDynamicState3ColorBlendEquation && eds3.extendedDynamicState3ColorWriteMask;
    dev.eds3_a2c = dev.dynamic_state3 && eds3.extendedDynamicState3AlphaToCoverageEnable;
    dev.vk12 = f12;
    dev.vk13 = f13;
    {   // (hookshot, after TotK Recomp's vk_caps.c) NVN's programs are written for 32 lanes: the translator is told 32, and the lane
        // shuffles of OoT's networks and the lane masks of its mesh programs depend on it. NVIDIA's GPUs have no other size; AMD's
        // default to 64 where they like, and the networks' output came out as coloured noise (Hookshot issues 10 and 14)
        VkPhysicalDeviceDriverProperties drv{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
        VkPhysicalDeviceSubgroupSizeControlProperties sc{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES};
        sc.pNext = &drv;
        VkPhysicalDeviceSubgroupProperties sg{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
        sg.pNext = &sc;
        VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        p2.pNext = &sg;
        vkGetPhysicalDeviceProperties2(dev.physical, &p2);
        const bool pin = f13.subgroupSizeControl && sc.minSubgroupSize <= 32 && sc.maxSubgroupSize >= 32 && !getenv("RX_VK_NO_SUBGROUP_PIN");
        dev.sg32_stages = pin ? sc.requiredSubgroupSizeStages : 0;
        dev.subgroup_size = sg.subgroupSize;
        dev.subgroup_max = sc.maxSubgroupSize > sg.subgroupSize ? sc.maxSubgroupSize : sg.subgroupSize;
        dev.driver_id = drv.driverID;
        logf("vk: subgroups: default %u lanes, %u-%u, pinned to 32 on stages %#x", sg.subgroupSize, sc.minSubgroupSize, sc.maxSubgroupSize,
             (unsigned)dev.sg32_stages);
    }
    if (!f13.dynamicRendering || !f13.synchronization2 || !f12.timelineSemaphore) {
        logf("vk: the GPU lacks dynamic rendering, synchronization2 or timeline semaphores");
        return false;
    }
    mesh.multiviewMeshShader = mesh.primitiveFragmentShadingRateMeshShader = VK_FALSE;   // (not used)
    dev.mesh_shader = dev.mesh_shader && mesh.meshShader && mesh.taskShader;
    // (task and mesh shader counts in pipeline statistics, a feature of their own: RADV has it off on RDNA2, and a query
    // pool made with those counts all the same can hang the GPU)
    dev.mesh_queries = dev.mesh_shader && mesh.meshShaderQueries;
    // (hookshot, issue 640) RDNA1 and older GPUs: no mesh shaders, but task and mesh programs run as compute shaders
    // that write their vertices to buffers, drawn through an ordinary graphics pipeline. It needs buffer device addresses.
    dev.mesh_fallback = !dev.mesh_shader && f12.bufferDeviceAddress && !getenv("RX_VK_NO_MESH_FALLBACK");
    if (dev.mesh_fallback) logf("vk: no mesh shaders: task and mesh programs run as compute shaders (issue 640 fallback)");
    bool task_mesh_fallback = false;
#ifdef _WIN32
    // Navi 22 can time out on task + mesh draws in the AMD Windows driver (issue 996).
    task_mesh_fallback = dev.props.vendorID == 0x1002 && dev.props.deviceID == 0x73df;
#endif
    if (const char* e = getenv("RX_VK_TASK_MESH_FALLBACK")) task_mesh_fallback = atoi(e) != 0;
    dev.task_mesh_fallback = task_mesh_fallback && dev.mesh_shader && f12.bufferDeviceAddress && !getenv("RX_VK_NO_MESH_FALLBACK");
    if (dev.task_mesh_fallback) logf("vk: task + mesh programs use the compute fallback (RX_VK_TASK_MESH_FALLBACK=0: native)");
    if (robust) logf("vk: robust buffer access %s", robust_ok ? "on" : "not supported");
    f.features.robustBufferAccess = robust_ok ? VK_TRUE : VK_FALSE;   // (off by default: the game's accesses stay in their buffers)
    dev.features = f.features;
    dev.sparse_images = (families[dev.queue_family].queueFlags & VK_QUEUE_SPARSE_BINDING_BIT) && f.features.sparseBinding &&
                        f.features.sparseResidencyImage2D && !(getenv("RX_VK_SPARSE") && !atoi(getenv("RX_VK_SPARSE")));
    dev.sparse_buffers = (families[dev.queue_family].queueFlags & VK_QUEUE_SPARSE_BINDING_BIT) && f.features.sparseBinding;
    VkDeviceCreateInfo ci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    ci.pNext = &f;
    ci.queueCreateInfoCount = compute_family != UINT32_MAX ? 2 : 1;
    ci.pQueueCreateInfos = qs;
    ci.enabledExtensionCount = (uint32_t)want.size();
    ci.ppEnabledExtensionNames = want.data();
    const VkResult dmade = vr::active() ? vr::create_device(dev.physical, &ci, &dev.device) : vkCreateDevice(dev.physical, &ci, nullptr, &dev.device);
    if (dmade != VK_SUCCESS) {
        logf("vk: vkCreateDevice failed");
        return false;
    }


    load_device(dev.device);
    vkGetDeviceQueue(dev.device, dev.queue_family, 0, &dev.queue);
    dev.present_queue = VK_NULL_HANDLE;
    dev.present_family = dev.queue_family;
    if (compute_family != UINT32_MAX) vkGetDeviceQueue(dev.device, compute_family, 0, &dev.present_queue), dev.present_family = compute_family;
    if (dev.buffer_marker) buffer_markers_init();
    coopmat_init();
    reflex_device_created();
    antilag_device_created();
    if (dev.push_descriptor) {
        VkPhysicalDevicePushDescriptorPropertiesKHR pp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PUSH_DESCRIPTOR_PROPERTIES_KHR};
        VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        p2.pNext = &pp;
        vkGetPhysicalDeviceProperties2(dev.physical, &p2);
        dev.max_push_descriptors = pp.maxPushDescriptors;
    }
    return true;
}

} // namespace

/// (hookshot) The UNORM format of an sRGB one (the same bytes, read and written as they are), else the format itself
VkFormat unorm_of(VkFormat f) {
    switch (f) {
    case VK_FORMAT_B8G8R8A8_SRGB: return VK_FORMAT_B8G8R8A8_UNORM;
    case VK_FORMAT_R8G8B8A8_SRGB: return VK_FORMAT_R8G8B8A8_UNORM;
    case VK_FORMAT_A8B8G8R8_SRGB_PACK32: return VK_FORMAT_A8B8G8R8_UNORM_PACK32;
    default: return f;
    }
}
/// (hookshot) The views of the swapchain's images let go (before the swapchain they view: the device idle already)
void swap_views_drop() {
    for (auto view : dev.swap_views)
        if (view) vkDestroyImageView(dev.device, view, nullptr);
    dev.swap_views.clear();
}
/// (hookshot) A view of each of the swapchain's images, for Dear ImGui's overlays (vk_imgui.cpp), when they can be drawn into,
/// in the format given (the swapchain's, or its UNORM one when the swapchain was made mutable)
void swap_views_make(VkImageUsageFlags usage, VkFormat format) {
    swap_views_drop();
    dev.swap_views.assign(dev.swap_images.size(), VK_NULL_HANDLE);
    dev.swap_view_format = format;
    if (!(usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)) return;
    for (size_t i = 0; i < dev.swap_images.size(); i++) {
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = dev.swap_images[i];
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = format;
        vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        if (vkCreateImageView(dev.device, &vi, nullptr, &dev.swap_views[i]) != VK_SUCCESS) dev.swap_views[i] = VK_NULL_HANDLE;
    }
}

bool create_swapchain(uint32_t width, uint32_t height) {
    VkSurfaceCapabilitiesKHR caps;
    VkResult result = vkGetPhysicalDeviceSurfaceCapabilitiesKHR(dev.physical, dev.surface, &caps);
    if (result != VK_SUCCESS) {
        logf("vk: querying surface capabilities failed (VkResult %d)", (int)result);
        return false;
    }
    if (caps.currentExtent.width != UINT32_MAX) width = caps.currentExtent.width, height = caps.currentExtent.height;
    logf("vk: surface caps images %u..%u extent %ux%u..%ux%u usage %#x alpha %#x transform %#x",
         caps.minImageCount, caps.maxImageCount, caps.minImageExtent.width, caps.minImageExtent.height,
         caps.maxImageExtent.width, caps.maxImageExtent.height, caps.supportedUsageFlags,
         caps.supportedCompositeAlpha, caps.supportedTransforms);
    if (!(caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT)) {
        logf("vk: surface does not support transfer-destination swapchain images");
        return false;
    }
    if (!width || !height) return false;
    uint32_t n = 0;
    result = vkGetPhysicalDeviceSurfaceFormatsKHR(dev.physical, dev.surface, &n, nullptr);
    if (result != VK_SUCCESS || !n) {
        logf("vk: querying surface formats failed (VkResult %d, count %u)", (int)result, n);
        return false;
    }
    std::vector<VkSurfaceFormatKHR> formats(n);
    vkGetPhysicalDeviceSurfaceFormatsKHR(dev.physical, dev.surface, &n, formats.data());
    VkSurfaceFormatKHR fmt = formats[0];
    for (const auto& f : formats)   // (an UNORM target: the game's window textures are presented as they are)
        // (in sRGB: RADV on Wayland lists every format again with PASS_THROUGH_EXT and the compositor's colour spaces,
        // and the last UNORM entry left the compositor an untagged surface, a grey picture. Still the last sRGB one, so
        // a driver that lists only sRGB (NVIDIA, X11, Windows) keeps the format it had.)
        if ((f.format == VK_FORMAT_B8G8R8A8_UNORM || f.format == VK_FORMAT_R8G8B8A8_UNORM) &&
            f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
            fmt = f;
    // (hookshot) the Insert menu's HDR: a 10-bit HDR10 surface when the display lists one (Windows: "Use HDR" on in its display
    // settings), the picture then written by framegen_hdr. Otherwise the SDR one above, said once
    dev.hdr = false;
    if (opts.hdr) {
        for (const auto& f : formats)
            if ((f.format == VK_FORMAT_A2B10G10R10_UNORM_PACK32 || f.format == VK_FORMAT_A2R10G10B10_UNORM_PACK32) &&
                f.colorSpace == VK_COLOR_SPACE_HDR10_ST2084_EXT)
                fmt = f, dev.hdr = true;
        static bool told;
        if (!told) {
            told = true;
            if (dev.hdr)
                logf("vk: HDR: an HDR10 surface (format %d), paper white %d nits, peak %d nits", (int)fmt.format, opts.hdr_white,
                     opts.hdr_peak);
            else
                logf("vk: HDR: the display lists no HDR10 surface%s: SDR",
                     dev.swapchain_colorspace ? " (is HDR on in the display settings?)" : " (no VK_EXT_swapchain_colorspace)");
        }
    }
    // mailbox when the driver has it (latest frame, no tear). The game paces itself (nn_vi.c).
    n = 0;
    result = vkGetPhysicalDeviceSurfacePresentModesKHR(dev.physical, dev.surface, &n, nullptr);
    std::vector<VkPresentModeKHR> modes;
    if (result == VK_SUCCESS && n) {
        modes.resize(n);
        result = vkGetPhysicalDeviceSurfacePresentModesKHR(dev.physical, dev.surface, &n, modes.data());
    }
    if (result != VK_SUCCESS || modes.empty())
        logf("vk: querying present modes failed (VkResult %d, count %u); trying mandatory FIFO mode", (int)result, n);
    VkPresentModeKHR mode = VK_PRESENT_MODE_FIFO_KHR;
    const bool force_fifo = getenv("RX_VK_FIFO") != nullptr;   // (as on a card or driver with vsync forced)
    if (!force_fifo) {
        for (auto m : modes)
            if (m == VK_PRESENT_MODE_MAILBOX_KHR) mode = m;
        // (Windows, no mailbox: AMD's driver has none, and FIFO held anything under 60 fps at 30.
        //  Linux: immediate on NVIDIA's X11 path turns vblank off for every output, so the other
        //  monitor and other programs tear. RX_VK_IMMEDIATE=1 still asks for it.)
        const bool want_immediate = getenv("RX_VK_IMMEDIATE") != nullptr;
#ifdef _WIN32
        if (want_immediate || mode == VK_PRESENT_MODE_FIFO_KHR)
#else
        if (want_immediate)
#endif
            for (auto m : modes)
                if (m == VK_PRESENT_MODE_IMMEDIATE_KHR) mode = m;
    }
    VkSwapchainCreateInfoKHR ci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    ci.surface = dev.surface;
    ci.minImageCount = std::max(caps.minImageCount, 4u);   // (hookshot: mailbox with frame interpolation: an image always free)
    if (caps.maxImageCount) ci.minImageCount = std::min(ci.minImageCount, caps.maxImageCount);
    ci.imageFormat = fmt.format;
    ci.imageColorSpace = fmt.colorSpace;
    ci.imageExtent = {width, height};
    ci.imageArrayLayers = 1;
    ci.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    // The renderer copies into swapchain images. Readback is optional and only available on surfaces that allow it.
    if (caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT)
        ci.imageUsage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    // (hookshot) the overlays Dear ImGui draws (vk_imgui.cpp) render into it. Every surface allows it (the Vulkan spec)
    if (caps.supportedUsageFlags & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)
        ci.imageUsage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    ci.preTransform = caps.currentTransform;
    ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    if (!(caps.supportedCompositeAlpha & ci.compositeAlpha)) {
        for (VkCompositeAlphaFlagBitsKHR alpha : {VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,
                                                  VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR,
                                                  VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR})
            if (caps.supportedCompositeAlpha & alpha) { ci.compositeAlpha = alpha; break; }
    }
    ci.presentMode = mode;
    const uint32_t sharing[2] = {dev.queue_family, dev.present_family};
    if (dev.present_queue && dev.present_family != dev.queue_family) {   // (presented from another family: no ownership moves)
        ci.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        ci.queueFamilyIndexCount = 2;
        ci.pQueueFamilyIndices = sharing;
    }
    ci.clipped = VK_TRUE;
    ci.oldSwapchain = dev.swapchain;
    // (hookshot) an sRGB swapchain (some drivers list no UNORM format) made mutable when the driver can: the overlays draw
    // through UNORM views, their colours as they are, as on a UNORM swapchain. Without it, vk_imgui.cpp makes them linear
    const VkFormat view_formats[2] = {fmt.format, unorm_of(fmt.format)};
    VkImageFormatListCreateInfo format_list{VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO};
    const bool mutable_view = dev.swap_mutable && view_formats[1] != fmt.format && (ci.imageUsage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
    if (mutable_view) {
        ci.flags |= VK_SWAPCHAIN_CREATE_MUTABLE_FORMAT_BIT_KHR;
        format_list.viewFormatCount = 2;
        format_list.pViewFormats = view_formats;
        format_list.pNext = ci.pNext;
        ci.pNext = &format_list;
    }
    reflex_swapchain_info(ci);   // (NVIDIA Reflex: latency mode on)
    VkSwapchainKHR sc;
    result = vkCreateSwapchainKHR(dev.device, &ci, nullptr, &sc);
    // Passing oldSwapchain retires it even when creation fails. Nothing may acquire from it again.
    if (dev.swapchain) {
        device_wait_idle_locked();   // (the present queue's presents of the old swapchain done too)
        reflex_swapchain_destroyed();   // (no latency sleep or markers may use the destroyed chain)
        swap_views_drop();   // (hookshot: the overlays' views of its images, before it)
        for (auto semaphore : dev.swap_rendered) vkDestroySemaphore(dev.device, semaphore, nullptr);
        vkDestroySwapchainKHR(dev.device, dev.swapchain, nullptr);
        dev.swapchain = VK_NULL_HANDLE;
        dev.swap_rendered.clear();
        dev.swap_images.clear();
        dev.swap_spare = 0;
    }
    if (result != VK_SUCCESS) {
        logf("vk: vkCreateSwapchainKHR failed (VkResult %d, extent %ux%u, format %d, usage %#x, mode %d, images %u)",
             (int)result, width, height, (int)fmt.format, ci.imageUsage, (int)mode, ci.minImageCount);
        return false;
    }
    if (vkGetSwapchainImagesKHR(dev.device, sc, &n, nullptr) != VK_SUCCESS || !n) {
        vkDestroySwapchainKHR(dev.device, sc, nullptr);
        return false;
    }
    std::vector<VkImage> images(n);
    if (vkGetSwapchainImagesKHR(dev.device, sc, &n, images.data()) != VK_SUCCESS || !n) {
        vkDestroySwapchainKHR(dev.device, sc, nullptr);
        return false;
    }
    images.resize(n);
    std::vector<VkSemaphore> rendered(n, VK_NULL_HANDLE);
    VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    for (auto& semaphore : rendered) {
        if (vkCreateSemaphore(dev.device, &si, nullptr, &semaphore) != VK_SUCCESS) {
            for (auto made : rendered)
                if (made) vkDestroySemaphore(dev.device, made, nullptr);
            vkDestroySwapchainKHR(dev.device, sc, nullptr);
            return false;
        }
    }
    dev.swapchain = sc;
    dev.swap_rendered = std::move(rendered);
    dev.swap_format = fmt.format;
    dev.swap_extent = {width, height};
    dev.swap_images = std::move(images);
    dev.swap_spare = n > caps.minImageCount ? n - caps.minImageCount : 0;
    logf("vk: swapchain %ux%u, %u images, %s", width, height, n,
         mode == VK_PRESENT_MODE_MAILBOX_KHR ? "mailbox" : mode == VK_PRESENT_MODE_IMMEDIATE_KHR ? "immediate" : "fifo");
    swap_views_make(ci.imageUsage, mutable_view ? view_formats[1] : fmt.format);
    reflex_swapchain_created(sc);   // (NVIDIA Reflex: its sleep mode on the new swapchain)
    return true;
}

namespace {

HWND main_window = nullptr;
// (hookshot) hookshot --prepare-shaders (a builder, after a build): the same work with no game, its progress as
// "@@progress" lines on stdout. A skip there leaves hookshot_prepare_skipped beside the game: its own starts then
// leave the work out too (what was not prepared is built when first drawn), until a --prepare-shaders run ends it
bool prepare_only() { return getenv("HOOKSHOT_PREPARE_ONLY") != nullptr; }
std::string skipped_path() {
    char exe[MAX_PATH];
    const DWORD n = game_module_path(exe, MAX_PATH);
    std::string path(exe, n);
    return path.substr(0, path.find_last_of("\\/") + 1) + "hookshot_prepare_skipped";
}
// (hookshot) The first start after an update prepares every shader (prepare_run) and the pipelines the last sessions noted
// (precompile_run): a minute and more, before the game's first frame. They run on a thread of their own while this one, the
// renderer's start, shows the window at once with what they are doing on it, and keeps it shown and sized; on Windows it is also
// the window's thread, and handles its messages meanwhile (it would not answer otherwise). Closing the window stops them
// (mkwvk_close_asked): what was made is kept, the next start goes on from there. RX_VK_STARTUP_SCREEN=0: as before, all
// of it before the window shows anything (and a console with the count on Windows)
std::atomic<int> prep_stage{0};   // 0 the programs, 1 the pipelines the notes predict (a first start), 2 the noted pipelines
std::atomic<size_t> prep_done{0}, prep_total{0};
std::atomic<bool> prep_running{false}, prep_finished{false};
HANDLE prep_ended = nullptr;
#ifndef _WIN32
const MkwVkWindow* prep_window = nullptr;   // (its size)
#endif
void prep_progress(size_t done, size_t total) {
    prep_done.store(done, std::memory_order_relaxed);
    prep_total.store(total, std::memory_order_relaxed);
}
bool screen_wanted() {
    const char* e = getenv("RX_VK_STARTUP_SCREEN");
    return !e || atoi(e) != 0;
}
/// The window's size now (0x0: minimized, or unknown)
void window_size(uint32_t* w, uint32_t* h) {
    *w = *h = 0;
#ifdef _WIN32
    RECT r{};
    if (main_window && GetClientRect(main_window, &r)) *w = (uint32_t)(r.right - r.left), *h = (uint32_t)(r.bottom - r.top);
#else
    int width = 0, height = 0;
    if (prep_window) prep_window->drawable_size(prep_window->window, &width, &height);
    if (width > 0 && height > 0) *w = (uint32_t)width, *h = (uint32_t)height;
#endif
}
void prepare_with_screen() {
    prep_ended = CreateEventA(nullptr, TRUE, FALSE, nullptr);
    prep_running.store(true);
    std::thread worker([] {
        const bool update = prepare_is_update();   // (an update, RX_VK_PREPARE_BACKGROUND=1: what the notes name now, the rest in play)
        const bool prepared = !update && prepare_run(prep_progress, false);
        if (prepared && !prepare_stopped()) {   // (a first start only: the pipelines the notes predict, into the driver's cache)
            prep_done.store(0), prep_total.store(0);
            prep_stage.store(1);
            predict_run(prep_progress);
        }
        if (!prepare_stopped()) {
            prep_done.store(0), prep_total.store(0);
            prep_stage.store(2);
            precompile_run(prep_progress);
        }
        if (prepare_stopped()) pipeline_cache_save(true);   // (the driver's part built so far, for the next start)
        else if (update) prepare_background();
        prep_finished.store(true, std::memory_order_release);
        SetEvent(prep_ended);
    });
    // (the screen is drawn and presented on a thread of its own: a present can wait for as long as the window cannot be seen (FIFO
    // on Wayland, a locked screen), and that must hold neither the window's messages nor the game's start)
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    static std::atomic<bool> paint_stop{false};
    HANDLE painted = CreateEventA(nullptr, TRUE, FALSE, nullptr);
    std::thread painter([t0, painted] {
        ProgressClock eta;
        int eta_stage = -1;
        double next_log = 10;
        bool prepared = false;   // (this start prepares the programs: a first start after an update, two steps)
        uint32_t last_w = dev.swap_extent.width, last_h = dev.swap_extent.height;
        for (;;) {
            const bool finished = prep_finished.load(std::memory_order_acquire);
            const double secs = std::chrono::duration<double>(clock::now() - t0).count();
            const int stage = prep_stage.load();
            const size_t done = prep_done.load(std::memory_order_relaxed), total = prep_total.load(std::memory_order_relaxed);
            if (stage != eta_stage) eta.reset(), eta_stage = stage;
            if (total) eta.note(secs, done, total);
            const double left = total ? eta.left() : -1;
            if (secs >= next_log && total) {
                logf("vk: start-up: %s %zu of %zu%s%s", stage == 2 ? "pipelines" : stage ? "predicted pipelines" : "shaders", done, total, left >= 0 ? ", " : "",
                     left_text(left).c_str());
                next_log = secs + 10;
            }
            if (stage == 0 && total) prepared = true;
            StartupState st;
            st.stage = stage, st.done = done, st.total = total, st.left = left, st.secs = secs;
            st.prepared = prepared, st.finished = finished, st.stopping = prepare_stopped();
            const StartupView view = startup_view(st);
            startup_screen(&view);
            uint32_t w, h;
            window_size(&w, &h);
            if (w && h && (w != last_w || h != last_h)) {   // (resized: the window's thread leaves it to the renderer, not started yet)
                dev.swap_resize.store((uint64_t)w << 32 | h, std::memory_order_release);
                last_w = w, last_h = h;
            }
            if (w && h) present_image(nullptr, dev.swap_extent.width, dev.swap_extent.height, nullptr);
            if (finished || paint_stop.load()) break;
            Sleep(33);
        }
        // (the end: a fade to black over 0.4 s, so the game's first frame comes out of black, not cut in over the screen; none
        // after a screen of under 1.5 s, a blink, or when the window is not shown (paint_stop: its presents wait))
        const double shown = std::chrono::duration<double>(clock::now() - t0).count();
        if (shown >= 1.5 && !paint_stop.load()) {
            const auto f0 = clock::now();
            for (;;) {
                const double f = std::chrono::duration<double>(clock::now() - f0).count() / 0.4;
                startup_fade_to((float)std::min(f, 1.0));
                uint32_t w, h;
                window_size(&w, &h);
                if (w && h) present_image(nullptr, dev.swap_extent.width, dev.swap_extent.height, nullptr);
                if (f >= 1 || paint_stop.load()) break;
                Sleep(8);
            }
        }
        SetEvent(painted);
    });
    // (this thread: the window's messages on Windows, until the work is done)
    for (;;) {
#ifdef _WIN32
        MSG m;
        while (PeekMessageA(&m, nullptr, 0, 0, PM_REMOVE)) {
            TranslateMessage(&m);
            DispatchMessageA(&m);
        }
#endif
        if (WaitForSingleObject(prep_ended, 33) == WAIT_OBJECT_0) break;
    }
    worker.join();
    // (the painter ends by itself once the work is finished, after its fade; it is told to stop only when it has not by then)
    if (WaitForSingleObject(painted, 1500) == WAIT_OBJECT_0) painter.join();
    else {
        paint_stop.store(true);   // (its present waits for the window to be seen: the game's own presents will, too)
        logf("vk: start-up: the window is not being shown; the game starts without waiting for it");
        painter.detach();
    }
    startup_screen(nullptr);
    prep_running.store(false);
    logf("vk: start-up: the shaders took %.1f s", std::chrono::duration<double>(clock::now() - t0).count());
}
bool started_up() {
    if (const char* e = getenv("RX_VK_START_DELAY")) Sleep((DWORD)atoi(e) * 1000);   // (diagnosis: a long start, as the first one is)
    if (!replay_start()) return false;
    if (prepare_only()) {
        // (three parts: the programs, the pipelines the notes predict, then those they name; most of the time is the first)
        static int part;
        const auto progress = [](size_t done, size_t total) {
            static size_t last = ~(size_t)0;
            if (!total) return;
            static const size_t from[3] = {0, 700, 900}, span[3] = {700, 200, 100};
            const size_t at = from[part] + done * span[part] / total;
            if (at == last) return;
            last = at;
            printf("@@progress %.3f\n", at / 1000.0);
            fflush(stdout);
        };
        printf("@@stage shaders Preparing shaders\n");
        fflush(stdout);
        part = 0;
        const bool prepared = prepare_run(progress, false);
        part = 1;
        if (prepared) predict_run(progress);
        part = 2;
        precompile_run(progress);
        pipeline_cache_save(true);
        remove(skipped_path().c_str());
        return true;
    }
    if (FILE* f = fopen(skipped_path().c_str(), "rb")) {
        fclose(f);
        logf("vk: the shaders' preparation was skipped at the build: left to when each is first drawn");
        return true;
    }
    if (precompile_wanted()) {   // (a release build, before the game: nothing to build during play)
        if (!dev.offscreen && screen_wanted()) {
            prepare_with_screen();
            return true;
        }
        // (a console while it lasts: there is no window yet, and a long wait with nothing on screen looks like a hang)
        static HANDLE console = nullptr;
        const auto progress = [](size_t done, size_t total) {
            static bool tried = false;
            if (!total) return;
            if (!tried) {
                tried = true;
                if (AllocConsole()) {
#ifdef _WIN32
                    // This console is ours. Selection must not block preparation's progress writes.
                    const HANDLE input = CreateFileA("CONIN$", GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
                                                     nullptr, OPEN_EXISTING, 0, nullptr);
                    if (input != INVALID_HANDLE_VALUE) {
                        DWORD mode;
                        if (GetConsoleMode(input, &mode))
                            SetConsoleMode(input, (mode & ~ENABLE_QUICK_EDIT_MODE) | ENABLE_EXTENDED_FLAGS);
                        CloseHandle(input);
                    }
#endif
                    SetConsoleTitleA("Hookshot");
                    console = CreateFileA("CONOUT$", GENERIC_WRITE, FILE_SHARE_WRITE, nullptr, OPEN_EXISTING, 0, nullptr);
                    if (console == INVALID_HANDLE_VALUE) console = nullptr;
                }
            }
            {   // (in the log too, every 10 s: a run with no window has nothing else to show it)
                static DWORD next_log = 0;
                const DWORD now = GetTickCount();
                if (!next_log) next_log = now + 10000;
                if ((int32_t)(now - next_log) >= 0) {
                    logf("vk: start-up: %zu of %zu", done, total);
                    next_log = now + 10000;
                }
            }
            if (!console) return;
            char line[80];
            const int n = snprintf(line, sizeof line, "%cPreparing shaders: %zu of %zu (%zu%%)   ", 13, done, total, done * 100 / total);
            DWORD wrote = 0;
            WriteFile(console, line, (DWORD)n, &wrote, nullptr);
        };
        const bool update = prepare_is_update();   // (an update: what the notes name now, the rest in play)
        if (!update && prepare_run(progress, false)) predict_run(progress);
        if (update) state_list_run();
        precompile_run(progress);
        if (update) prepare_background();
        if (console) CloseHandle(console), console = nullptr, FreeConsole();
    }
    return true;
}
// (the window's surface made: the presents' own queue kept where its family can present to it, create_device)
static void choose_present_queue() {
    if (!dev.present_queue) return;
    VkBool32 ok = VK_FALSE;
    if (vkGetPhysicalDeviceSurfaceSupportKHR(dev.physical, dev.present_family, dev.surface, &ok) != VK_SUCCESS || !ok) {
        logf("vk: queue family %u cannot present to the window: presents on the replay's queue", dev.present_family);
        dev.present_queue = VK_NULL_HANDLE, dev.present_family = dev.queue_family;
        return;
    }
    logf("vk: presents on a queue of their own, of compute family %u (RX_VK_PRESENT_QUEUE=0: off)", dev.present_family);
}

int start(void* hwnd) {
    main_window = (HWND)hwnd;
    if (!hwnd) {   // (HOS_HEADLESS: no window; the frames are rendered all the same, RX_VK_SHOTS writes them out)
        dev.offscreen = true;
        dev.swap_extent = {1920, 1080};
        dev.swap_format = VK_FORMAT_R8G8B8A8_UNORM;
        logf("vk: no window: presents render offscreen (%ux%u)", dev.swap_extent.width, dev.swap_extent.height);
        if (!started_up()) return 0;
        pipeline_cache_play();   // (play begins: what is made from now on goes into the play files)
        return 1;
    }
#ifdef _WIN32
    VkWin32SurfaceCreateInfoKHR si{VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR};
    si.hinstance = GetModuleHandleA(nullptr);
    si.hwnd = (HWND)hwnd;
    if (vkCreateWin32SurfaceKHR(dev.instance, &si, nullptr, &dev.surface) != VK_SUCCESS) return 0;
    VkBool32 ok = VK_FALSE;
    vkGetPhysicalDeviceSurfaceSupportKHR(dev.physical, dev.queue_family, dev.surface, &ok);
    if (!ok) return 0;
    choose_present_queue();
    RECT r{};
    GetClientRect((HWND)hwnd, &r);
    if (!create_swapchain((uint32_t)(r.right - r.left), (uint32_t)(r.bottom - r.top))) return 0;
#else
    const MkwVkWindow* w = (const MkwVkWindow*)hwnd;
    prep_window = w;
    if (w->version < 1 || w->version > MKWVK_WINDOW_VERSION) {   // (1: no refresh_mhz)
        logf("vk: unsupported SDL window bridge version %u (expected %u)", w->version, MKWVK_WINDOW_VERSION);
        return 0;
    }
    if (!w->create_surface(w->window, (void*)dev.instance, (void*)&dev.surface)) {
        logf("vk: SDL_Vulkan_CreateSurface failed; check SDL_VIDEODRIVER and the Vulkan window-system extensions");
        return 0;
    }
    VkBool32 ok = VK_FALSE;
    VkResult present = vkGetPhysicalDeviceSurfaceSupportKHR(dev.physical, dev.queue_family, dev.surface, &ok);
    if (present != VK_SUCCESS || !ok) {
        logf("vk: queue-family present support failed (VkResult %d, supported %u)", (int)present, ok);
        return 0;
    }
    choose_present_queue();
    int width = 0, height = 0;
    w->drawable_size(w->window, &width, &height);
    if (width <= 0 || height <= 0) {
        logf("vk: SDL returned zero drawable size %dx%d", width, height);
        return 0;
    }
    if (!create_swapchain((uint32_t)width, (uint32_t)height)) {
        logf("vk: swapchain creation failed for drawable %dx%d", width, height);
        return 0;
    }
#endif
    if (!started_up()) {
        logf("vk: renderer startup failed after window and swapchain initialization");
        return 0;
    }
    pipeline_cache_play();   // (play begins: what is made from now on goes into the play files)
    return 1;
}

// (the window's thread: only noted. The swapchain is remade by whoever presents next, in present_image: made here, it went from under a
// frame between its acquire and its submit, and the frame's commands named images of the swapchain just destroyed: a lost device)
void resize(int width, int height) {
    if (dev.offscreen || width <= 0 || height <= 0) return;
    dev.swap_resize.store((uint64_t)(uint32_t)width << 32 | (uint32_t)height, std::memory_order_release);
}

const char* describe() {
    static std::string d;
    d = std::string(dev.props.deviceName) + " (Vulkan, native)";
    return d.c_str();
}

MkwVkRenderer table;

} // namespace
} // namespace mkwvk

// (hookshot) The window is being closed (host/display.c, host/posix/display_sdl.c, before they end the process): while the
// shaders are prepared, their workers are stopped and what they made is kept first (10 s at most: a driver can take longer on one
// pipeline). Looked up by name, so an executable and a back end of different versions still close
extern "C"
#ifdef _WIN32
__declspec(dllexport)
#else
__attribute__((visibility("default")))
#endif
void mkwvk_close_asked(void) {
    using namespace mkwvk;
    pipeline_cache_close();   // (in play: the pipelines made since the last save, kept for the next start)
    if (!prep_running.load() || prepare_stopped()) return;
    logf("vk: the window was closed while the shaders were prepared: stopping, what is done is kept");
    prepare_stop();
    if (WaitForSingleObject(prep_ended, 10000) != WAIT_OBJECT_0) logf("vk: the preparation did not stop within 10 s: closing anyway");
}

extern "C"
#ifdef _WIN32
__declspec(dllexport)
#else
__attribute__((visibility("default")))
#endif
const MkwVkRenderer* mkwvk_init(const MkwVkHost* h) {
    using namespace mkwvk;
    if (!h || h->version != MKWVK_API_VERSION) return nullptr;
    host = h;
    if (!load_global()) {
        logf("vk: vulkan-1.dll not found");
        return nullptr;
    }
    logf("vr: build %s; HOOKSHOT_VR=%s",
#ifdef HOOKSHOT_VR_BUILD
         "has VR",
#else
         "has NO VR",
#endif
         getenv("HOOKSHOT_VR") ? getenv("HOOKSHOT_VR") : "(not set)");
    if (vr::wanted() && !vr::init()) logf("vr: HOOKSHOT_VR is set but VR could not start: the game runs on the desktop only");
    if (!create_instance() || !pick_device() || !create_device()) return nullptr;
    if (vr::active() && !vr::create_session()) logf("vr: the OpenXR session could not be made: the game runs on the desktop only");
    logf("vk: %s, Vulkan %u.%u, driver %#x; mesh shaders %s, host memory import %s, pipeline libraries %s",
         dev.props.deviceName, VK_VERSION_MAJOR(dev.props.apiVersion), VK_VERSION_MINOR(dev.props.apiVersion),
         dev.props.driverVersion, dev.mesh_shader ? "yes" : "no", dev.external_memory_host ? "yes" : "no",
         dev.graphics_pipeline_library ? "yes" : "no");
    if (!dev.mesh_shader && !dev.mesh_fallback && !getenv("RX_VK_NO_MESH_WARNING")) {
        // (the game draws its foliage, grass and many effects with mesh shaders: without them those draws are left out and the
        // picture is broken (an RX 580's "static"-like scenes, 2026-11-08). Said once, on its own thread, and the game goes on.)
        // (a GPU with no buffer device addresses has no fallback to run either)
        logf("vk: this GPU has no mesh shaders: the game's foliage, grass and many effects cannot be drawn");
#ifdef _WIN32
        std::thread([name = std::string(dev.props.deviceName)] {
            const std::string text = "Your graphics card (" + name + ") or its driver has no mesh shader support, which Ocarina of Time "
                "uses for foliage, grass and many effects, so parts of the picture will be missing or broken.\n\nHookshot needs a GPU "
                "with mesh shaders: NVIDIA GeForce RTX 20 series or newer, AMD Radeon RX 6000 series or newer, or Intel Arc. If your "
                "card is one of those, update its driver.";
            MessageBoxA(nullptr, text.c_str(), "Hookshot: graphics card", MB_OK | MB_ICONWARNING | MB_TOPMOST);
        }).detach();
#endif
    }
    table.version = MKWVK_API_VERSION;
    table.start = start;
    table.resize = resize;
    table.describe = describe;
    fill_replay_table(table);
    return &table;
}

namespace mkwvk {
HWND app_window() { return main_window; }   // (hookshot: the Insert menu's Fullscreen row)

/// (hookshot) The refresh rate of the display the window is on, in mHz; 0 when not known (frame generation's Auto). Asked
/// again once a second at most: the window may move to another display, and Windows' answer costs a few system calls
int display_refresh_mhz() {
    static std::atomic<int> cached{0};
    static std::atomic<int64_t> asked_at{0};
    const int64_t now = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    const int64_t last = asked_at.load(std::memory_order_relaxed);
    if (last && now - last < 1000) return cached.load(std::memory_order_relaxed);
    asked_at.store(now, std::memory_order_relaxed);
    int mhz = 0;
#ifdef _WIN32
    MONITORINFOEXW mi{};
    mi.cbSize = sizeof mi;
    DEVMODEW dm{};
    dm.dmSize = sizeof dm;
    if (main_window && GetMonitorInfoW(MonitorFromWindow(main_window, MONITOR_DEFAULTTONEAREST), &mi) &&
        EnumDisplaySettingsW(mi.szDevice, ENUM_CURRENT_SETTINGS, &dm) && dm.dmDisplayFrequency > 1)   // (0 and 1: the default)
        // (whole Hz: a 59.94 Hz mode often says 59, and Auto would then make a frame too few a second. One under a
        // multiple of 6 (24, 30, 60, 120, 144, 240 Hz) is taken as that rate's /1.001)
        mhz = (dm.dmDisplayFrequency + 1) % 6 == 0 ? (int)((dm.dmDisplayFrequency + 1) * 1000000ull / 1001)
                                                    : (int)dm.dmDisplayFrequency * 1000;
#else
    const MkwVkWindow* w = (const MkwVkWindow*)main_window;
    if (w && w->version >= 2 && w->refresh_mhz) mhz = w->refresh_mhz(w->window);
#endif
    cached.store(mhz, std::memory_order_relaxed);
    return mhz;
}
} // namespace mkwvk
