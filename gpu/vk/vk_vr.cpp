// VR cinema mode for Hookshot: see vk_vr.h. NOT YET BUILT OR RUN by its author: the first build may need small fixes.
//
// How it fits the renderer:
//  - vk_main.cpp: when HOOKSHOT_VR=1 the Vulkan instance, the GPU and the device are made through OpenXR
//    (XR_KHR_vulkan_enable2: xrCreateVulkanInstanceKHR / xrGetVulkanGraphicsDevice2KHR / xrCreateVulkanDeviceKHR), so the
//    runtime adds the extensions it needs and picks the GPU the headset is on. Hookshot's own create infos pass through.
//  - vk_replay.cpp present_image: calls present() first. present() copies the game's picture into an OpenXR swapchain image
//    with a command buffer and fence of its own, submitted on the same queue (the queue lock is dev.present_mu).
//  - The window keeps presenting as before (a mirror of the game on the desktop).
#ifdef HOOKSHOT_VR_BUILD

#include "vk_vr.h"
#include "vk_internal.h"
#include "vk_state.h"

#define XR_NO_PROTOTYPES
#define XR_USE_GRAPHICS_API_VULKAN
#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <map>
#include <set>
#include <chrono>
#include <memory>
#include <mutex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace mkwvk {
namespace vr {
namespace {

// ---- the OpenXR functions, from openxr_loader.dll (no import library) --------------------------------------------------
PFN_xrGetInstanceProcAddr xrGetInstanceProcAddr_;
#define XRX_GLOBAL(X) X(xrEnumerateInstanceExtensionProperties) X(xrCreateInstance)
#define XRX_INSTANCE(X)                                                                                                    \
    X(xrDestroyInstance) X(xrResultToString) X(xrGetSystem) X(xrGetVulkanGraphicsRequirements2KHR)                         \
    X(xrCreateVulkanInstanceKHR) X(xrGetVulkanGraphicsDevice2KHR) X(xrCreateVulkanDeviceKHR) X(xrCreateSession)           \
    X(xrDestroySession) X(xrCreateReferenceSpace) X(xrDestroySpace) X(xrEnumerateSwapchainFormats) X(xrCreateSwapchain)    \
    X(xrDestroySwapchain) X(xrEnumerateSwapchainImages) X(xrAcquireSwapchainImage) X(xrWaitSwapchainImage)                 \
    X(xrReleaseSwapchainImage) X(xrPollEvent) X(xrBeginSession) X(xrEndSession) X(xrWaitFrame) X(xrBeginFrame) X(xrEndFrame) X(xrLocateViews)
#define XRX_DECL(f) PFN_##f f;
XRX_GLOBAL(XRX_DECL)
XRX_INSTANCE(XRX_DECL)
#undef XRX_DECL

struct Slot {
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;   // signalled when the slot's last copy is done
};

struct State {
    HMODULE dll = nullptr;
    XrInstance instance = XR_NULL_HANDLE;
    XrSystemId system = XR_NULL_SYSTEM_ID;
    XrSession session = XR_NULL_HANDLE;
    XrSpace space = XR_NULL_HANDLE;
    bool created = false, running = false, dead = false;

    XrSwapchain sc = XR_NULL_HANDLE;
    uint32_t w = 0, h = 0;
    int64_t format = 0;
    bool raw_copy = false;
    std::vector<VkImage> images;
    std::vector<int64_t> formats;

    Slot slots[3];
    uint32_t slot = 0;

    float distance = 2.5f, width = 3.2f;   // metres: how far the screen is, and how wide
    int64_t frames = 0;
};
State g;

bool chk(XrResult r, const char* what) {
    if (XR_SUCCEEDED(r)) return true;
    char s[XR_MAX_RESULT_STRING_SIZE] = "?";
    if (g.instance && xrResultToString) xrResultToString(g.instance, r, s);
    mkwvk::logf("vr: %s failed: %s (%d)", what, s, (int)r);
    return false;
}
#define XR_CHK(call) chk((call), #call)

float env_float(const char* name, float def) {
    const char* e = getenv(name);
    return (e && *e) ? (float)atof(e) : def;
}

bool load_runtime() {
    g.dll = LoadLibraryA("openxr_loader.dll");   // (beside hookshot.exe, or on the PATH)
    if (!g.dll) {
        mkwvk::logf("vr: openxr_loader.dll not found: put it beside hookshot.exe (VR is off)");
        return false;
    }
    xrGetInstanceProcAddr_ = (PFN_xrGetInstanceProcAddr)(void*)GetProcAddress(g.dll, "xrGetInstanceProcAddr");
    if (!xrGetInstanceProcAddr_) return false;
#define XRX_LOAD(f) xrGetInstanceProcAddr_(XR_NULL_HANDLE, #f, (PFN_xrVoidFunction*)&f);
    XRX_GLOBAL(XRX_LOAD)
#undef XRX_LOAD
    return xrCreateInstance && xrEnumerateInstanceExtensionProperties;
}

void load_instance_functions() {
#define XRX_LOAD(f) xrGetInstanceProcAddr_(g.instance, #f, (PFN_xrVoidFunction*)&f);
    XRX_INSTANCE(XRX_LOAD)
#undef XRX_LOAD
}

// ---- the screen: one OpenXR swapchain the size of the game's picture ------------------------------------------------------
void wait_slots() {
    for (Slot& s : g.slots)
        if (s.fence) vkWaitForFences(dev.device, 1, &s.fence, VK_TRUE, UINT64_MAX);
}

void destroy_swapchain() {
    if (!g.sc) return;
    wait_slots();
    std::lock_guard qk(dev.present_mu);
    xrDestroySwapchain(g.sc);
    g.sc = XR_NULL_HANDLE;
    g.images.clear();
}

bool pick_format() {
    if (g.formats.empty()) {
        uint32_t n = 0;
        if (!XR_CHK(xrEnumerateSwapchainFormats(g.session, 0, &n, nullptr))) return false;
        g.formats.resize(n);
        if (!XR_CHK(xrEnumerateSwapchainFormats(g.session, n, &n, g.formats.data()))) return false;
    }
    {
        std::string all;
        for (int64_t f : g.formats) all += std::to_string((long long)f) + " ";
        mkwvk::logf("vr: the runtime offers swapchain formats: %s", all.c_str());
    }
    auto offered = [&](VkFormat f) {
        for (int64_t s : g.formats)
            if (s == (int64_t)f) return true;
        return false;
    };
    // First choice UNORM: the game's picture goes to the window as it is, so it is blitted here the same way.
    // Failing that (SteamVR often offers only SRGB formats): the same bytes are copied raw into an SRGB image, which the
    // headset reads back as the colours the game made. A blit would encode them a second time and wash the picture out.
    g.format = 0;
    g.raw_copy = false;
    for (VkFormat f : {VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM})
        if (offered(f)) { g.format = (int64_t)f; break; }
    if (!g.format) {
        // (the channel order of the game's picture: the window's, unless HOOKSHOT_VR_SWAPRB=1 turns red and blue round)
        const bool bgr_src = dev.swap_format == VK_FORMAT_B8G8R8A8_UNORM || dev.swap_format == VK_FORMAT_B8G8R8A8_SRGB;
        const char* flip = getenv("HOOKSHOT_VR_SWAPRB");
        const bool bgr = (flip && *flip && *flip != '0') ? !bgr_src : bgr_src;
        const VkFormat order[2] = {bgr ? VK_FORMAT_B8G8R8A8_SRGB : VK_FORMAT_R8G8B8A8_SRGB,
                                   bgr ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_B8G8R8A8_SRGB};
        for (VkFormat f : order)
            if (offered(f)) { g.format = (int64_t)f; g.raw_copy = true; break; }
    }
    if (!g.format) {
        mkwvk::logf("vr: the runtime offers no 8-bit swapchain format I can use");
        return false;
    }
    return true;
}

bool ensure_swapchain(uint32_t w, uint32_t h) {
    if (g.sc && g.w == w && g.h == h) return true;
    destroy_swapchain();
    if (!pick_format()) return false;
    XrSwapchainCreateInfo ci{XR_TYPE_SWAPCHAIN_CREATE_INFO};
    ci.usageFlags = XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT | XR_SWAPCHAIN_USAGE_TRANSFER_DST_BIT | XR_SWAPCHAIN_USAGE_SAMPLED_BIT;
    ci.format = g.format;
    ci.sampleCount = 1;
    ci.width = w;
    ci.height = h;
    ci.faceCount = 1;
    ci.arraySize = 1;
    ci.mipCount = 1;
    {
        std::lock_guard qk(dev.present_mu);
        if (!XR_CHK(xrCreateSwapchain(g.session, &ci, &g.sc))) return false;
    }
    uint32_t n = 0;
    if (!XR_CHK(xrEnumerateSwapchainImages(g.sc, 0, &n, nullptr))) return false;
    std::vector<XrSwapchainImageVulkan2KHR> imgs(n, XrSwapchainImageVulkan2KHR{XR_TYPE_SWAPCHAIN_IMAGE_VULKAN2_KHR});
    if (!XR_CHK(xrEnumerateSwapchainImages(g.sc, n, &n, (XrSwapchainImageBaseHeader*)imgs.data()))) return false;
    g.images.clear();
    for (auto& i : imgs) g.images.push_back(i.image);
    g.w = w;
    g.h = h;
    mkwvk::logf("vr: screen %ux%u, %u images, format %lld", w, h, n, (long long)g.format);
    return true;
}

bool make_slots() {
    for (Slot& s : g.slots) {
        VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pi.queueFamilyIndex = dev.queue_family;
        if (vkCreateCommandPool(dev.device, &pi, nullptr, &s.pool) != VK_SUCCESS) return false;
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool = s.pool;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(dev.device, &ai, &s.cmd) != VK_SUCCESS) return false;
        VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        if (vkCreateFence(dev.device, &fi, nullptr, &s.fence) != VK_SUCCESS) return false;
    }
    return true;
}

// ---- session events (the queue's lock is held: xrBeginSession and xrEndSession may use the queue) -------------------------
void poll_events() {
    std::lock_guard qk(dev.present_mu);
    XrEventDataBuffer ev{XR_TYPE_EVENT_DATA_BUFFER};
    while (xrPollEvent(g.instance, &ev) == XR_SUCCESS) {
        if (ev.type == XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED) {
            const auto* e = (const XrEventDataSessionStateChanged*)&ev;
            switch (e->state) {
            case XR_SESSION_STATE_READY: {
                XrSessionBeginInfo bi{XR_TYPE_SESSION_BEGIN_INFO};
                bi.primaryViewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
                if (XR_CHK(xrBeginSession(g.session, &bi))) {
                    g.running = true;
                    mkwvk::logf("vr: session running");
                }
                break;
            }
            case XR_SESSION_STATE_STOPPING:
                xrEndSession(g.session);
                g.running = false;
                mkwvk::logf("vr: session stopped");
                break;
            case XR_SESSION_STATE_EXITING:
            case XR_SESSION_STATE_LOSS_PENDING:
                g.running = false;
                g.dead = true;   // (the user quit from SteamVR: the game goes on in its window)
                mkwvk::logf("vr: session ended by the runtime; the game continues on the desktop");
                break;
            default: break;
            }
        } else if (ev.type == XR_TYPE_EVENT_DATA_INSTANCE_LOSS_PENDING) {
            g.running = false;
            g.dead = true;
        }
        ev = XrEventDataBuffer{XR_TYPE_EVENT_DATA_BUFFER};
    }
}

void barrier(VkCommandBuffer cmd, VkImage image, VkImageLayout from, VkImageLayout to, VkPipelineStageFlags2 srcStage,
             VkAccessFlags2 srcAccess, VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess) {
    VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    b.srcStageMask = srcStage;
    b.srcAccessMask = srcAccess;
    b.dstStageMask = dstStage;
    b.dstAccessMask = dstAccess;
    b.oldLayout = from;
    b.newLayout = to;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkDependencyInfo d{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    d.imageMemoryBarrierCount = 1;
    d.pImageMemoryBarriers = &b;
    vkCmdPipelineBarrier2(cmd, &d);
}

void end_frame_empty(XrTime t) {
    std::lock_guard qk(dev.present_mu);
    XrFrameEndInfo ei{XR_TYPE_FRAME_END_INFO};
    ei.displayTime = t;
    ei.environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;
    XR_CHK(xrEndFrame(g.session, &ei));
}

} // namespace

// =====================================================================================================================
bool wanted() {
    const char* e = getenv("HOOKSHOT_VR");
    return e && *e && *e != '0';
}

bool active() { return g.instance != XR_NULL_HANDLE && !g.dead; }

bool init() {
    if (!load_runtime()) return false;

    uint32_t n = 0;
    xrEnumerateInstanceExtensionProperties(nullptr, 0, &n, nullptr);
    std::vector<XrExtensionProperties> props(n, XrExtensionProperties{XR_TYPE_EXTENSION_PROPERTIES});
    xrEnumerateInstanceExtensionProperties(nullptr, n, &n, props.data());
    bool has_vk2 = false;
    for (auto& p : props)
        if (!strcmp(p.extensionName, XR_KHR_VULKAN_ENABLE2_EXTENSION_NAME)) has_vk2 = true;
    if (!has_vk2) {
        mkwvk::logf("vr: the OpenXR runtime has no XR_KHR_vulkan_enable2: is SteamVR the active OpenXR runtime? (VR is off)");
        return false;
    }
    const char* exts[] = {XR_KHR_VULKAN_ENABLE2_EXTENSION_NAME};
    XrInstanceCreateInfo ci{XR_TYPE_INSTANCE_CREATE_INFO};
    snprintf(ci.applicationInfo.applicationName, XR_MAX_APPLICATION_NAME_SIZE, "Hookshot");
    snprintf(ci.applicationInfo.engineName, XR_MAX_ENGINE_NAME_SIZE, "Hookshot VR");
    ci.applicationInfo.applicationVersion = 1;
    ci.applicationInfo.engineVersion = 1;
    ci.applicationInfo.apiVersion = XR_CURRENT_API_VERSION;
    ci.enabledExtensionCount = 1;
    ci.enabledExtensionNames = exts;
    XrResult cr = xrCreateInstance(&ci, &g.instance);
    if (cr == XR_ERROR_API_VERSION_UNSUPPORTED) {   // (the runtime is older than the headers: ask for OpenXR 1.0, which all of them have)
        mkwvk::logf("vr: the OpenXR runtime does not take the newest API version: asking for 1.0");
        ci.applicationInfo.apiVersion = ((uint64_t)1) << 48;
        cr = xrCreateInstance(&ci, &g.instance);
    }
    if (XR_FAILED(cr)) {
        mkwvk::logf("vr: xrCreateInstance failed (%d)", (int)cr);
        g.instance = XR_NULL_HANDLE;
        return false;
    }
    load_instance_functions();

    XrSystemGetInfo gi{XR_TYPE_SYSTEM_GET_INFO};
    gi.formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;
    if (!XR_CHK(xrGetSystem(g.instance, &gi, &g.system))) {
        mkwvk::logf("vr: no headset found: start SteamVR with the Quest connected before the game (VR is off)");
        xrDestroyInstance(g.instance);
        g.instance = XR_NULL_HANDLE;
        return false;
    }
    // (must be asked before the Vulkan instance is made)
    XrGraphicsRequirementsVulkan2KHR req{XR_TYPE_GRAPHICS_REQUIREMENTS_VULKAN2_KHR};
    if (!XR_CHK(xrGetVulkanGraphicsRequirements2KHR(g.instance, g.system, &req))) return false;
    mkwvk::logf("vr: runtime takes Vulkan %u.%u to %u.%u (Hookshot asks for 1.3)", (unsigned)XR_VERSION_MAJOR(req.minApiVersionSupported),
                (unsigned)XR_VERSION_MINOR(req.minApiVersionSupported), (unsigned)XR_VERSION_MAJOR(req.maxApiVersionSupported),
                (unsigned)XR_VERSION_MINOR(req.maxApiVersionSupported));
    g.distance = env_float("HOOKSHOT_VR_DISTANCE", 2.5f);
    g.width = env_float("HOOKSHOT_VR_WIDTH", 3.2f);
    return true;
}

VkResult create_instance(const VkInstanceCreateInfo* info, VkInstance* out) {
    XrVulkanInstanceCreateInfoKHR ci{XR_TYPE_VULKAN_INSTANCE_CREATE_INFO_KHR};
    ci.systemId = g.system;
    ci.pfnGetInstanceProcAddr = vkx::instance_proc_addr();
    ci.vulkanCreateInfo = info;
    ci.vulkanAllocator = nullptr;
    VkResult vr = VK_SUCCESS;
    if (!XR_CHK(xrCreateVulkanInstanceKHR(g.instance, &ci, out, &vr))) return VK_ERROR_INITIALIZATION_FAILED;
    return vr;
}

VkPhysicalDevice physical_device(VkInstance instance) {
    XrVulkanGraphicsDeviceGetInfoKHR gi{XR_TYPE_VULKAN_GRAPHICS_DEVICE_GET_INFO_KHR};
    gi.systemId = g.system;
    gi.vulkanInstance = instance;
    VkPhysicalDevice pd = VK_NULL_HANDLE;
    if (!XR_CHK(xrGetVulkanGraphicsDevice2KHR(g.instance, &gi, &pd))) return VK_NULL_HANDLE;
    return pd;
}

VkResult create_device(VkPhysicalDevice phys, const VkDeviceCreateInfo* info, VkDevice* out) {
    XrVulkanDeviceCreateInfoKHR ci{XR_TYPE_VULKAN_DEVICE_CREATE_INFO_KHR};
    ci.systemId = g.system;
    ci.pfnGetInstanceProcAddr = vkx::instance_proc_addr();
    ci.vulkanPhysicalDevice = phys;
    ci.vulkanCreateInfo = info;
    ci.vulkanAllocator = nullptr;
    VkResult vr = VK_SUCCESS;
    if (!XR_CHK(xrCreateVulkanDeviceKHR(g.instance, &ci, out, &vr))) return VK_ERROR_INITIALIZATION_FAILED;
    return vr;
}

void stereo_init();
bool stereo_take_handled();

bool create_session() {
    XrGraphicsBindingVulkan2KHR gb{XR_TYPE_GRAPHICS_BINDING_VULKAN2_KHR};
    gb.instance = dev.instance;
    gb.physicalDevice = dev.physical;
    gb.device = dev.device;
    gb.queueFamilyIndex = dev.queue_family;
    gb.queueIndex = 0;   // (the one queue Hookshot makes: its locks are the ones this module takes, too)
    XrSessionCreateInfo si{XR_TYPE_SESSION_CREATE_INFO};
    si.next = &gb;
    si.systemId = g.system;
    if (!XR_CHK(xrCreateSession(g.instance, &si, &g.session))) return false;

    XrReferenceSpaceCreateInfo rs{XR_TYPE_REFERENCE_SPACE_CREATE_INFO};
    // HOOKSHOT_VR_HEADLOCK=1: the screen follows the head. Default: it stays where the headset was when the game started.
    const char* hl = getenv("HOOKSHOT_VR_HEADLOCK");
    rs.referenceSpaceType = (hl && *hl && *hl != '0') ? XR_REFERENCE_SPACE_TYPE_VIEW : XR_REFERENCE_SPACE_TYPE_LOCAL;
    rs.poseInReferenceSpace.orientation.w = 1.0f;
    if (!XR_CHK(xrCreateReferenceSpace(g.session, &rs, &g.space))) return false;
    if (!make_slots()) {
        mkwvk::logf("vr: command buffers for the copy failed");
        return false;
    }
    g.created = true;
    stereo_init();
    mkwvk::logf("vr: OpenXR session made; the game's picture will show on a %.1f m wide screen %.1f m away", g.width, g.distance);
    return true;
}

void present(VkImage src, uint32_t sw, uint32_t sh, const int crop[4]) {
    if (!g.created || g.dead || !src) return;
    if (stereo_take_handled()) return;   // (the frame already went to the headset as two eyes)
    poll_events();
    if (!g.running) return;

    XrFrameState fs{XR_TYPE_FRAME_STATE};
    XrFrameWaitInfo wi{XR_TYPE_FRAME_WAIT_INFO};
    if (!XR_CHK(xrWaitFrame(g.session, &wi, &fs))) return;   // (paces this thread to the headset)
    {
        std::lock_guard qk(dev.present_mu);
        XrFrameBeginInfo bi{XR_TYPE_FRAME_BEGIN_INFO};
        if (!XR_CHK(xrBeginFrame(g.session, &bi))) return;
    }
    if (!fs.shouldRender) {
        end_frame_empty(fs.predictedDisplayTime);
        return;
    }

    // the part of the picture shown: the crop the window's present uses
    int x0 = 0, y0 = 0, x1 = (int)sw, y1 = (int)sh;
    if (crop && crop[2] > 0 && crop[3] > 0) x0 = crop[0], y0 = crop[1], x1 = std::min(crop[0] + crop[2], (int)sw), y1 = std::min(crop[1] + crop[3], (int)sh);
    const uint32_t cw = (uint32_t)std::max(1, x1 - x0), ch = (uint32_t)std::max(1, y1 - y0);

    if (!ensure_swapchain(cw, ch)) {
        mkwvk::logf("vr: the screen could not be made: VR is off");
        g.dead = true;
        end_frame_empty(fs.predictedDisplayTime);
        return;
    }
    uint32_t index = 0;
    XrSwapchainImageAcquireInfo ai{XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO};
    {
        std::lock_guard qk(dev.present_mu);
        if (!XR_CHK(xrAcquireSwapchainImage(g.sc, &ai, &index))) {
            XrFrameEndInfo ei{XR_TYPE_FRAME_END_INFO};
            ei.displayTime = fs.predictedDisplayTime;
            ei.environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;
            xrEndFrame(g.session, &ei);
            return;
        }
    }
    XrSwapchainImageWaitInfo wsi{XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO};
    wsi.timeout = XR_INFINITE_DURATION;
    const bool waited = XR_CHK(xrWaitSwapchainImage(g.sc, &wsi));

    bool submitted = false;
    if (waited && index < g.images.size()) {
        Slot& s = g.slots[g.slot];
        g.slot = (g.slot + 1) % 3;
        vkWaitForFences(dev.device, 1, &s.fence, VK_TRUE, UINT64_MAX);
        vkResetCommandPool(dev.device, s.pool, 0);
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(s.cmd, &bi);

        // (everything the game's earlier submits wrote, like present_image's full barrier before its own blit)
        VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        mb.srcStageMask = mb.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        mb.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
        VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dep.memoryBarrierCount = 1;
        dep.pMemoryBarriers = &mb;
        vkCmdPipelineBarrier2(s.cmd, &dep);

        VkImage dst = g.images[index];
        barrier(s.cmd, dst, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0,
                VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
        VkImageBlit b{};
        b.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        b.srcOffsets[0] = {x0, y0, 0};
        b.srcOffsets[1] = {x1, y1, 1};
        b.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        b.dstOffsets[1] = {(int32_t)cw, (int32_t)ch, 1};
        if (g.raw_copy) {
            VkImageCopy c{};
            c.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            c.srcOffset = {x0, y0, 0};
            c.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            c.extent = {cw, ch, 1};
            vkCmdCopyImage(s.cmd, src, VK_IMAGE_LAYOUT_GENERAL, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &c);
        } else {
            vkCmdBlitImage(s.cmd, src, VK_IMAGE_LAYOUT_GENERAL, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &b, VK_FILTER_LINEAR);
        }
        barrier(s.cmd, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
        vkEndCommandBuffer(s.cmd);

        VkCommandBufferSubmitInfo cb{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
        cb.commandBuffer = s.cmd;
        VkSubmitInfo2 si{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
        si.commandBufferInfoCount = 1;
        si.pCommandBufferInfos = &cb;
        std::lock_guard qk(dev.present_mu);
        queue_flush_locked();   // (held-back game submits out first, as present_image does)
        vkResetFences(dev.device, 1, &s.fence);
        if (vkQueueSubmit2(dev.queue, 1, &si, s.fence) == VK_SUCCESS) {
            submitted = true;
        } else {
            mkwvk::logf("vr: the copy's submit failed: VR is off");
            g.dead = true;
            vkQueueSubmit2(dev.queue, 0, nullptr, s.fence);   // (leaves the fence signalled again, so no wait hangs)
        }
    }

    // (the copy was submitted before the image goes back: OpenXR's rule. The queue's lock covers both calls)
    std::lock_guard qk(dev.present_mu);
    XrSwapchainImageReleaseInfo ri{XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO};
    xrReleaseSwapchainImage(g.sc, &ri);

    XrCompositionLayerQuad quad{XR_TYPE_COMPOSITION_LAYER_QUAD};
    quad.layerFlags = 0;
    quad.space = g.space;
    quad.eyeVisibility = XR_EYE_VISIBILITY_BOTH;
    quad.subImage.swapchain = g.sc;
    quad.subImage.imageRect.offset = {0, 0};
    quad.subImage.imageRect.extent = {(int32_t)g.w, (int32_t)g.h};
    quad.subImage.imageArrayIndex = 0;
    quad.pose.orientation.w = 1.0f;
    quad.pose.position = {0.0f, 0.0f, -g.distance};
    quad.size.width = g.width;
    quad.size.height = g.width * (float)g.h / (float)g.w;
    const XrCompositionLayerBaseHeader* layers[1] = {(const XrCompositionLayerBaseHeader*)&quad};

    XrFrameEndInfo ei{XR_TYPE_FRAME_END_INFO};
    ei.displayTime = fs.predictedDisplayTime;
    ei.environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;
    ei.layerCount = submitted ? 1 : 0;
    ei.layers = submitted ? layers : nullptr;
    XR_CHK(xrEndFrame(g.session, &ei));
    if (++g.frames == 1) mkwvk::logf("vr: first frame sent to the headset");
}

// ---- stereo screen (HOOKSHOT_VR_STEREO=1) ------------------------------------------------------------------------------------
// The game's frame is replayed (vk_replay.cpp stereo_eyes) once for each eye with its camera changed: the view moved sideways by
// half the eye separation and the projection sheared so that things at the convergence distance land at the same place in both
// eyes. The game keeps its camera in the uniform data in several shapes (the world-to-view and view-to-world matrices as 3x4
// rows, the same as 4x4 columns, the projection, and their product); patch_uniform finds each by its values and writes the eye's.
// Both eyes' pictures go to the headset as two quad layers, one visible to each eye, on the screen of the cinema mode.
namespace {

struct Cam {
    bool valid = false;
    float V[16] = {}, C[16] = {}, P[16] = {};   // 4x4, row by row: world to view, view to world, projection
};
struct Eyes {   // a frame's camera, and what it becomes for each eye (and the last frame's, for what the game keeps of it)
    Cam game, prev;
    float V[2][16], C[2][16], P[2][16], VP[2][16];
    float pV[2][16], pC[2][16], pP[2][16], pVP[2][16];
    bool has_prev = false;
    float sep = 0;   // game units: what these were made with
    bool has_pose = false;   // head tracking: E[k] is eye k's pose in the game camera's frame (metres), raw[k] the headset's own (x y z, quaternion x y z w)
    float E[2][16];
    float raw[2][7];
};
struct Stereo {
    bool on = false;
    std::atomic<int> eye{-1};
    std::atomic<bool> have{false};
    std::mutex mu;
    std::shared_ptr<const Eyes> eyes;
    Cam last;
    float sep = 6.4f, conv = 190.f;   // game units: the eyes' distance apart, and where the two pictures meet
    bool sep_fixed = false;           // (HOOKSHOT_VR_SEPARATION: not made from the headset's eye distance)
    float ipd = 0.064f;               // metres, from the headset
    bool left_done = false;           // the game's own replay was the left eye's
    std::vector<std::pair<uint64_t, uint64_t>> hints;   // where the camera was found (address, bytes): looked at first next time
    XrTime period = 0;
    std::chrono::steady_clock::time_point last_end{};
    double ms_sum = 0;
    XrSwapchain sc[2] = {XR_NULL_HANDLE, XR_NULL_HANDLE};
    std::vector<VkImage> images[2];
    uint32_t w = 0, h = 0;
    XrTime time = 0;
    bool copied[2] = {false, false};
    bool handled = false;
    int64_t counts[14] = {};   // V, C, P, product, transposed, rotation-only view, rotation-only inverse, bare position: changed in all
    int64_t frames = 0;
    // safety: stereo only runs while an "arm" file exists, and pauses itself when the game hiccups (loading) or runs slow
    bool go = false;                     // decided once per game frame (stereo_frame_start)
    int64_t tick = 0, hold_until = 0, slow = 0;
    std::chrono::steady_clock::time_point last_start{};
    bool armed = false;
    bool announced = false;
    std::atomic<bool> scan_req{false};   // hookshot_vr_scan.txt: look for the camera in the game's own memory (once)
    bool fov_started = false;
    // head tracking: the eyes' poses from the headset (in its LOCAL space), and the same relative to where the head was when stereo started
    bool pose_ok = false, recenter = true;
    float raw[2][7] = {};
    float Erel[2][16] = {};
    float A_inv[16] = {};
    float A[16] = {};
    float yaw0 = 0.f;
    float lastE[2][16] = {};
    bool last_pose = false;
    XrPosef cpose[2] = {};      // the pose each cached picture was drawn from
    bool cpose_ok[2] = {false, false};
    long long pose_log = 0;
    std::atomic<bool> dump_req{false};   // hookshot_vr_dump.txt: write both eye pictures to hookshot_vr_eyes.png (once)
    float cfx[2] = {1.2f, 1.2f}, cfy[2] = {2.1f, 2.1f}, csep[2] = {0.f, 0.f};   // (the field of view and separation each cached picture was drawn with)
    float last_logged_sx = 0.f;
    int stale = 0;                       // frames in a row sent from the cache (no fresh picture)
    int64_t nfresh = 0, nstale = 0;
    int aer_eye = 1;                     // alternate-eye rendering: the eye this game frame is drawn for
    std::atomic<bool> patched{false};    // this frame's draws really got an eye's camera
    VkImage cimg[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};   // each eye's latest picture (the other eye's is from the frame before)
    VkDeviceMemory cmem[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    bool cvalid[2] = {false, false};
    uint32_t cw = 0, ch = 0;
    int cidx = 0, clog = 0;              // the second eye's compute passes: this frame's number, and how many were logged
    std::mutex smu;
    std::vector<std::pair<int, int>> skips;   // numbers to leave out (hookshot_vr_skip.txt)
    bool skip_all = false;
};
Stereo st;

void mul4(const float* a, const float* b, float* o) {
    float t[16];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) {
            float v = 0;
            for (int k = 0; k < 4; k++) v += a[i * 4 + k] * b[k * 4 + j];
            t[i * 4 + j] = v;
        }
    std::memcpy(o, t, 64);
}
void transpose4(const float* a, float* o) {
    float t[16];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) t[j * 4 + i] = a[i * 4 + j];
    std::memcpy(o, t, 64);
}
bool nearly(float a, float b, float tol) { return std::fabs(a - b) <= tol * (1.f + std::fabs(a) + std::fabs(b)); }

bool rigid3(const float* r) {   // three rows of four: a rotation in the first three columns, a position in the fourth
    for (int i = 0; i < 12; i++)
        if (!(std::fabs(r[i]) < 1e7f)) return false;
    for (int i = 0; i < 3; i++) {
        const float* v = r + i * 4;
        if (!nearly(v[0] * v[0] + v[1] * v[1] + v[2] * v[2], 1.f, 2e-3f)) return false;
        for (int k = i + 1; k < 3; k++) {
            const float* u = r + k * 4;
            if (std::fabs(v[0] * u[0] + v[1] * u[1] + v[2] * u[2]) > 4e-3f) return false;
        }
    }
    const bool ident = nearly(r[0], 1.f, 1e-4f) && nearly(r[5], 1.f, 1e-4f) && nearly(r[10], 1.f, 1e-4f) && std::fabs(r[1]) < 1e-4f &&
                       std::fabs(r[2]) < 1e-4f && std::fabs(r[4]) < 1e-4f && std::fabs(r[6]) < 1e-4f && std::fabs(r[8]) < 1e-4f &&
                       std::fabs(r[9]) < 1e-4f;
    return !ident;
}
bool is_proj(const float* a) {   // a perspective as the game makes it: [sx 0 0 0; 0 sy 0 0; 0 0 A B; 0 0 -1 0]
    return a[1] == 0 && a[2] == 0 && a[3] == 0 && a[4] == 0 && a[6] == 0 && a[7] == 0 && a[8] == 0 && a[9] == 0 && a[12] == 0 &&
           a[13] == 0 && a[15] == 0 && a[14] == -1.f && a[0] > 0 && a[5] > 0 && a[10] < -0.9f && a[10] > -1.5f && a[11] < 0;
}

// the camera in these bytes: a projection, and the world-to-view and view-to-world matrices that are each other's inverse
bool find_camera(const float* f, size_t nf, Cam& out) {
    long p = -1;
    for (size_t o = 0; o + 16 <= nf; o += 4)
        if (is_proj(f + o)) {   // (the picture's camera is wider than tall: not a square probe or cube-map projection)
            const float q = f[o + 5] / f[o];
            if (q < 1.3f || q > 3.8f) continue;
            p = (long)o;
            break;
        }
    if (p < 0) return false;
    std::vector<size_t> tr;
    for (size_t o = 0; o + 12 <= nf && tr.size() < 32; o += 4)
        if (rigid3(f + o)) tr.push_back(o);
    for (size_t a : tr)
        for (size_t b : tr) {
            if (a == b) continue;
            const float* A = f + a;   // world to view: x' = R x + t
            const float* B = f + b;   // view to world: R^T, and -R^T t
            bool ok = true;
            for (int i = 0; i < 3 && ok; i++) {
                for (int j = 0; j < 3; j++)
                    if (std::fabs(B[i * 4 + j] - A[j * 4 + i]) > 2e-3f) { ok = false; break; }
                float c = 0;
                for (int j = 0; j < 3; j++) c -= A[j * 4 + i] * A[j * 4 + 3];
                if (!nearly(B[i * 4 + 3], c, 1e-2f)) ok = false;
            }
            if (!ok) continue;
            out = Cam{};
            std::memcpy(out.V, A, 48), out.V[15] = 1.f;
            std::memcpy(out.C, B, 48), out.C[15] = 1.f;
            std::memcpy(out.P, f + p, 64);
            out.valid = true;
            return true;
        }
    return false;
}


// ---- head tracking ----------------------------------------------------------------------------------------------------------
void quat_to_mat(float x, float y, float z, float w, float* R /* 3x3 row-major */) {
    R[0] = 1 - 2 * (y * y + z * z), R[1] = 2 * (x * y - z * w), R[2] = 2 * (x * z + y * w);
    R[3] = 2 * (x * y + z * w), R[4] = 1 - 2 * (x * x + z * z), R[5] = 2 * (y * z - x * w);
    R[6] = 2 * (x * z - y * w), R[7] = 2 * (y * z + x * w), R[8] = 1 - 2 * (x * x + y * y);
}
void pose_to_mat4(const float* p /* x y z qx qy qz qw */, float* M) {
    float R[9];
    quat_to_mat(p[3], p[4], p[5], p[6], R);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) M[i * 4 + j] = R[i * 3 + j];
        M[i * 4 + 3] = p[i];
    }
    M[12] = M[13] = M[14] = 0.f, M[15] = 1.f;
}
void rigid_inv4(const float* E, float* o) {
    float t[16];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) t[i * 4 + j] = E[j * 4 + i];
        float v = 0;
        for (int j = 0; j < 3; j++) v -= E[j * 4 + i] * E[j * 4 + 3];
        t[i * 4 + 3] = v;
    }
    t[12] = t[13] = t[14] = 0.f, t[15] = 1.f;
    std::memcpy(o, t, 64);
}
// HOOKSHOT_VR_VFOV (degrees, default 90) and HOOKSHOT_VR_HFOV (degrees, default: the game's own): the window each eye's picture is drawn
// through. The game's own projection is only 50 degrees tall; the headset sees more, so the picture is drawn with its own projection.
void headset_window(float* P, const float* game) {   // (game: the game's own projection: its shape is kept, or the HUD drawn on the picture is squeezed)
    static const float vf = [] { const char* e = getenv("HOOKSHOT_VR_VFOV"); return e && *e ? (float)atof(e) : 90.f; }();
    static const float hf = [] { const char* e = getenv("HOOKSHOT_VR_HFOV"); return e && *e ? (float)atof(e) : 0.f; }();
    if (vf > 20.f && vf < 175.f) {
        P[5] = 1.f / std::tan(vf * 0.5f * 0.0174533f);
        if (game[5] > 0.f) P[0] = P[5] * (game[0] / game[5]);   // (the same width to height as the game's)
    }
    if (hf > 20.f && hf < 175.f) P[0] = 1.f / std::tan(hf * 0.5f * 0.0174533f);
}
// the camera moved to the eye's pose (in the camera's own frame): the world to the eye is the inverse of that, after the camera's own
void eye_of_pose(const Cam& c, const float* E, float* V, float* C, float* P) {
    float Ei[16];
    rigid_inv4(E, Ei);
    mul4(Ei, c.V, V);
    mul4(c.C, E, C);
    std::memcpy(P, c.P, 64);
    headset_window(P, c.P);
}
bool head_tracking_enabled() {
    static const bool on = [] { const char* e = getenv("HOOKSHOT_VR_HEADTRACK"); return !(e && *e == '0'); }();
    return on;
}

void eye_of(const Cam& c, float sepx, float* V, float* C, float* P) {   // the camera moved sideways by sepx, its view sheared
    std::memcpy(V, c.V, 64), std::memcpy(C, c.C, 64), std::memcpy(P, c.P, 64);
    V[3] -= sepx;                                              // (view x' = x - sepx)
    for (int i = 0; i < 3; i++) C[i * 4 + 3] += sepx * C[i * 4];   // (the camera's position moves along its right axis)
    static const bool offaxis = [] { const char* e = getenv("HOOKSHOT_VR_OFFAXIS"); return e && *e && *e != '0'; }();
    if (offaxis) P[2] -= c.P[0] * sepx / st.conv;              // (off-axis: zero parallax at the convergence distance. Off by default: the game's depth/fog maths use the projection)
}

// ---- the game's own camera, turned by the head (HOOKSHOT_VR_CAMCTL=1) ------------------------------------------------------------
// The game works out what to draw, the light's direction and its shadows from its own camera. Turning only the picture after that leaves
// those behind: objects pop and shadows swim. So the game's camera is turned itself, as the game's native mouse look does (host/
// mouse_native.c): the player camera's controller holds its pitch limits {-85, 89, -85, 89} (degrees), its distance after them, its pitch
// at +0x14 and its yaw at +0x2c (degrees; yaw falls as the view turns right, pitch rises as it tilts down). Each frame the head's turn
// since the last frame is added to those. What the picture still needs (the head's roll, its position) is drawn as before, and the
// pose the picture was drawn from is what the headset is told.
namespace {
struct CamHist {
    int64_t tick = -1;
    bool driven = false;
    float E[2][16];     // the eyes' poses relative to where the head started (the straight-ahead frame)
    float raw[2][7];    // the same in the headset's own space
    float Rp[16];       // the head's yaw and pitch alone: what the game's camera was turned by
};
struct CamCtl {
    bool enabled = false;
    int lag = 2;                       // (frames between a head pose being pushed and the picture drawn from it)
    float dist = -1.f;                 // HOOKSHOT_VR_CAMDIST: the camera's distance from the player (a test), negative: left alone
    std::atomic<float> live_dist{0.f}; // the distance the game's camera holds now (its controller's +0x10), read each frame
    std::vector<uint8_t*> ctl;         // the controllers found (host pointers to their limits)
    std::vector<uint8_t*> found;
    std::atomic<bool> searching{false}, answered{false};
    std::chrono::steady_clock::time_point next_search{};
    int64_t wait_ms = 3000;
    bool have_last = false;
    double last_th = 0, last_ph = 0;
    long long pushes = 0;
    float ysign = -1.f, psign = 1.f;   // HOOKSHOT_VR_CAMYAW / HOOKSHOT_VR_CAMPITCH: 1 or -1, which way the game's angles follow the head
    double pushed_th = 0, pushed_ph = 0;   // the head angles the game's camera has been given so far
    float pcmd = 0.f; bool have_pcmd = false;   // the pitch the game's camera was last told to have (written again every frame: the game eases its pitch back to centre by itself)
    CamHist hist[16];
};
CamCtl cc;
const float cam_sig[4] = {-85.0f, 89.0f, -85.0f, 89.0f};
float cam_getf(const uint8_t* p) { float f; std::memcpy(&f, p, 4); return f; }
void cam_putf(uint8_t* p, float f) { std::memcpy(p, &f, 4); }
bool cam_angles_ok(const uint8_t* p) {
    const float pt = cam_getf(p + 0x14), y = cam_getf(p + 0x2c);
    return std::isfinite(pt) && std::isfinite(y) && std::fabs(pt) <= 90.f && std::fabs(y) <= 360.f;
}
bool cam_valid(const uint8_t* p) { return !std::memcmp(p, cam_sig, sizeof cam_sig) && cam_angles_ok(p); }
#ifdef _WIN32
struct WsInfo { void* va; ULONG_PTR attr; };
typedef BOOL(WINAPI* QwsFn)(HANDLE, void*, DWORD);
void cam_search_thread() {
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<uint8_t*> out;
    uint8_t* gb = host ? host->guest_base() : nullptr;
    const uint64_t gs = host ? host->guest_size() : 0, HB = 0x1000000000ULL;   // (the guest heap)
    if (gb && gs > HB) {
        const uint64_t span = std::min<uint64_t>(gs - HB, 12ull << 30), pages = span / 4096;
        std::vector<uint8_t> use((size_t)pages, 1);
        QwsFn qws = (QwsFn)GetProcAddress(GetModuleHandleA("kernel32.dll"), "K32QueryWorkingSetEx");
        if (qws) {   // (only the pages the process has touched)
            const uint64_t B = 65536;
            std::vector<WsInfo> info((size_t)B);
            for (uint64_t i = 0; i < pages; i += B) {
                const uint64_t n = std::min(B, pages - i);
                for (uint64_t k = 0; k < n; k++) info[(size_t)k].va = gb + HB + (i + k) * 4096, info[(size_t)k].attr = 0;
                if (qws(GetCurrentProcess(), info.data(), (DWORD)(n * sizeof(WsInfo))))
                    for (uint64_t k = 0; k < n; k++) use[(size_t)(i + k)] = (uint8_t)(info[(size_t)k].attr & 1);
            }
        }
        uint32_t first;
        std::memcpy(&first, &cam_sig[0], 4);
        for (uint64_t i = 0; i < pages && out.size() < 8; i++) {
            if (!use[(size_t)i]) continue;
            uint8_t* p = gb + HB + i * 4096;
            for (size_t o = 0; o + 16 <= 4096 && out.size() < 8; o += 4)
                if (*(const uint32_t*)(p + o) == first && !std::memcmp(p + o, cam_sig, sizeof cam_sig) && cam_angles_ok(p + o)) out.push_back(p + o);
            if ((i & 4095) == 4095) Sleep(1);
        }
    }
    mkwvk::logf("vr-cam: %zu camera controller(s) found in %lld ms", out.size(),
                (long long)std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count());
    cc.found = out;
    cc.answered = true;
    cc.searching = false;
}
#endif
size_t cam_live() {
    size_t n = 0;
    for (uint8_t* p : cc.ctl) n += cam_valid(p) ? 1 : 0;
    return n;
}
void cam_service() {   // (each frame: take up a search's answer; look again if none is live)
    if (cc.answered.exchange(false)) {
        cc.ctl = cc.found;
        cc.wait_ms = cc.ctl.empty() ? std::min<int64_t>(cc.wait_ms * 2, 30000) : 3000;
        cc.next_search = std::chrono::steady_clock::now() + std::chrono::milliseconds(cc.wait_ms);
    }
#ifdef _WIN32
    if (cam_live() == 0 && !cc.searching.load() && std::chrono::steady_clock::now() >= cc.next_search) {
        cc.searching = true;
        cc.next_search = std::chrono::steady_clock::now() + std::chrono::milliseconds(cc.wait_ms);
        std::thread(cam_search_thread).detach();
    }
#endif
}
// (a test, HOOKSHOT_VR_CAMPOKE="4c=0.5,58=0.5": numbers written into the camera controller at these offsets (hex) every frame)
void cam_poke(uint8_t* p) {
    static const std::vector<std::pair<int, float>> pokes = [] {
        std::vector<std::pair<int, float>> v;
        const char* e = getenv("HOOKSHOT_VR_CAMPOKE");
        std::string t = e ? e : "";
        size_t i = 0;
        while (i < t.size()) {
            const size_t j = t.find(',', i), end = j == std::string::npos ? t.size() : j;
            const std::string item = t.substr(i, end - i);
            const size_t eq = item.find('=');
            if (eq != std::string::npos) v.push_back({(int)std::strtol(item.substr(0, eq).c_str(), nullptr, 16), (float)atof(item.c_str() + eq + 1)});
            i = end + 1;
        }
        return v;
    }();
    for (const auto& q : pokes)
        if (q.first >= 0 && q.first < 0x400) cam_putf(p + q.first, q.second);
}
void cam_push(double dth, double dph) {   // radians: the head turned left by dth and up by dph
    for (uint8_t* p : cc.ctl) {
        if (!cam_valid(p)) continue;
        if (!cc.have_pcmd) cc.pcmd = cam_getf(p + 0x14), cc.have_pcmd = true;
        float y = cam_getf(p + 0x2c) + (float)(cc.ysign * dth * 57.29578), pt = cc.pcmd - (float)(cc.psign * dph * 57.29578);
        if (y > 180.f) y -= 360.f;
        if (y < -180.f) y += 360.f;
        const float lo = cam_getf(p), hi = cam_getf(p + 4);
        if (pt < lo) pt = lo;
        if (pt > hi) pt = hi;
        cc.pcmd = pt;
        cam_putf(p + 0x2c, y), cam_putf(p + 0x14, pt);
        if (cc.dist >= 0.f) cam_putf(p + 0x10, cc.dist);
        cam_poke(p);
        {   // (a test, HOOKSHOT_VR_CAMDUMP=1: the controller's numbers, three times, to find the one that sets the camera's distance)
            static const bool dump = [] { const char* e = getenv("HOOKSHOT_VR_CAMDUMP"); return e && *e && *e != '0'; }();
            static int calls = 0, dumps = 0;
            if (dump && dumps < 4 && (calls++ % 240) == 120) {
                dumps++;
                for (int row = 0; row < 32; row++) {
                    float v[8];
                    for (int i = 0; i < 8; i++) v[i] = cam_getf(p + row * 32 + i * 4);
                    mkwvk::logf("vr-cam-dump %d: +0x%03x: %g %g %g %g %g %g %g %g", dumps, row * 32, (double)v[0], (double)v[1], (double)v[2], (double)v[3],
                                (double)v[4], (double)v[5], (double)v[6], (double)v[7]);
                }
            }
        }
    }
}
void cam_hold_pitch() {   // every frame: the pitch stays where the head put it
    if (!cc.have_pcmd) return;
    for (uint8_t* p : cc.ctl)
        if (cam_valid(p)) cam_putf(p + 0x14, cc.pcmd), cam_poke(p);
}
const CamHist* cam_hist(int64_t tick) {   // the entry for this tick (or the nearest older one), if the game's camera was driven then
    for (int back = 0; back < 8; back++) {
        const CamHist& h = cc.hist[(tick - back) & 15];
        if (h.tick == tick - back) return h.driven ? &h : nullptr;
    }
    return nullptr;
}
}

// ---- finding which draws make something (a test): hookshot_vr_hide.txt leaves draws out; hookshot_vr_hidelist.txt lists the programs ----
// hookshot_vr_hide.txt holds: ind (every indirect draw: the GPU decides how many), inst (every instanced draw), and numbers or ranges
// (3, 10-20) of the programs in the last list; nothing or no file: everything is drawn. The list is written to the log.
namespace {
struct HideInfo { uint64_t n = 0, ind = 0, inst = 0; };
std::mutex hide_mu;
std::map<uint64_t, HideInfo> hide_seen;
std::vector<uint64_t> hide_snap;
std::set<uint64_t> hide_keys;
bool hide_ind = false, hide_inst = false;
std::atomic<bool> hide_any{false};
void hide_poll(int64_t tick) {
    if ((tick % 30) == 11) {   // (the list)
        if (FILE* f = fopen("hookshot_vr_hidelist.txt", "rb")) {
            fclose(f);
            remove("hookshot_vr_hidelist.txt");
            std::lock_guard lk(hide_mu);
            hide_snap.clear();
            for (auto& kv : hide_seen) hide_snap.push_back(kv.first);
            mkwvk::logf("vr-draws: %zu programs; draws, indirect, instanced since the last list:", hide_snap.size());
            for (size_t i = 0; i < hide_snap.size() && i < 700; i++) {
                const HideInfo& h = hide_seen[hide_snap[i]];
                mkwvk::logf("vr-draws: #%zu prog %016llx draws %llu indirect %llu instanced %llu", i, (unsigned long long)hide_snap[i],
                            (unsigned long long)h.n, (unsigned long long)h.ind, (unsigned long long)h.inst);
            }
            for (auto& kv : hide_seen) kv.second = HideInfo{};
        }
    }
    if ((tick % 30) == 9) {   // (what to leave out)
        bool ind = false, inst = false;
        std::set<uint64_t> keys;
        if (FILE* f = fopen("hookshot_vr_hide.txt", "rb")) {
            char buf[2048] = {};
            const size_t n = fread(buf, 1, sizeof buf - 1, f);
            fclose(f);
            buf[n] = 0;
            std::lock_guard lk(hide_mu);
            for (char* p = strtok(buf, " ,;\r\n\t"); p; p = strtok(nullptr, " ,;\r\n\t")) {
                if (!strcmp(p, "ind")) ind = true;
                else if (!strcmp(p, "inst")) inst = true;
                else if (*p >= '0' && *p <= '9') {
                    size_t a = (size_t)atoi(p), b = a;
                    if (const char* d = strchr(p, '-')) b = (size_t)atoi(d + 1);
                    for (size_t i = a; i <= b && i < hide_snap.size(); i++) keys.insert(hide_snap[i]);
                }
            }
        }
        std::lock_guard lk(hide_mu);
        if (ind != hide_ind || inst != hide_inst || keys != hide_keys) {
            hide_ind = ind, hide_inst = inst, hide_keys = keys;
            hide_any = ind || inst || !keys.empty();
            mkwvk::logf("vr-draws: hiding %s%s%zu program(s)", ind ? "indirect draws, " : "", inst ? "instanced draws, " : "", keys.size());
        }
    }
}
}

// HOOKSHOT_VR_FPDIST=2: the distance is measured from how the game's camera moves. With the player still, the camera goes round his head on a
// sphere: when the view turns by df (a change in its direction), the camera moves by dc = -D * df, and D is the distance. Each game frame's
// pair of cameras gives one measure, kept when dc is that and not the player walking; it is blended in, and kept while the view is still.
float fp_est = 0.f;
bool fp_est_have = false;
void fp_measure(const Cam& cur, const Cam& last) {
    if (!last.valid || !cur.valid) return;
    const float df[3] = {-(cur.C[2] - last.C[2]), -(cur.C[6] - last.C[6]), -(cur.C[10] - last.C[10])};
    const float dc[3] = {cur.C[3] - last.C[3], cur.C[7] - last.C[7], cur.C[11] - last.C[11]};
    const float ff = df[0] * df[0] + df[1] * df[1] + df[2] * df[2];
    if (ff < 4e-6f) return;   // (the view hardly turned: nothing to measure by)
    const float D = -(dc[0] * df[0] + dc[1] * df[1] + dc[2] * df[2]) / ff;
    if (!(D > 0.5f && D < 30.f)) return;
    const float r0 = dc[0] + D * df[0], r1 = dc[1] + D * df[1], r2 = dc[2] + D * df[2];
    const float res = std::sqrt(r0 * r0 + r1 * r1 + r2 * r2), mag = std::sqrt(dc[0] * dc[0] + dc[1] * dc[1] + dc[2] * dc[2]);
    if (res > 0.25f * mag + 0.01f) return;   // (the camera moved some other way as well: the player walking)
    const float w = std::min(1.f, std::sqrt(ff) / 0.02f) * 0.35f;
    fp_est = fp_est_have ? fp_est + w * (D - fp_est) : D;
    fp_est_have = true;
    static int n = 0;
    if ((++n % 200) == 1) mkwvk::logf("vr-cam: first person: measured camera distance %.2f (this pair %.2f; the game's field %.2f)", (double)fp_est, (double)D, (double)cc.live_dist.load());
}
// (HOOKSHOT_VR_FPDIST=1: the eyes go forward by the distance the game's camera holds now, less HOOKSHOT_VR_BACK metres (default 0.2); 2: by the
// measured distance; otherwise by HOOKSHOT_VR_FWD metres)
float fp_fwd() {
    static const float v = [] { const char* e = getenv("HOOKSHOT_VR_FWD"); return e && *e ? (float)atof(e) : 0.f; }();
    static const int mode = [] { const char* e = getenv("HOOKSHOT_VR_FPDIST"); return e && *e ? atoi(e) : 0; }();
    static const float back = [] { const char* e = getenv("HOOKSHOT_VR_BACK"); return e && *e ? (float)atof(e) : 0.2f; }();
    if (mode == 2) {
        const float d = fp_est_have ? fp_est : cc.live_dist.load();
        if (d > 0.3f) return std::max(0.f, d - back);
    } else if (mode == 1) {
        const float d = cc.live_dist.load();
        if (d > 0.3f) return std::max(0.f, d - back);
    }
    return v;
}
float fp_up() {
    static const float v = [] { const char* e = getenv("HOOKSHOT_VR_UP"); return e && *e ? (float)atof(e) : 0.f; }();
    return v;
}
std::shared_ptr<Eyes> make_eyes(const Cam& cur, const Cam& last) {
    fp_measure(cur, last);
    auto e = std::make_shared<Eyes>();
    e->game = cur;
    e->sep = st.sep;
    if (last.valid) e->prev = last, e->has_prev = true;
    for (int k = 0; k < 2; k++) {
        const float sepx = (k == 0 ? -0.5f : 0.5f) * st.sep;
        if (st.pose_ok) {
            e->has_pose = true;
            const CamHist* h = cc.enabled ? cam_hist(st.tick - cc.lag) : nullptr;
            // (HOOKSHOT_VR_OFF containing c, a test: both eyes use the left eye's pose, so the camera does not jump sideways every frame)
            static const bool cyclops = [] { const char* o = getenv("HOOKSHOT_VR_OFF"); return o && strchr(o, 'c'); }();
            const int kk = cyclops ? 0 : k;
            if (h) {   // (the game's camera was turned by the head: the picture needs only what is left, and says which pose it was drawn from)
                float Rpi[16];
                rigid_inv4(h->Rp, Rpi);
                mul4(Rpi, h->E[kk], e->E[k]);
                std::memcpy(e->raw[k], h->raw[kk], sizeof e->raw[k]);
            } else {
                std::memcpy(e->E[k], st.Erel[kk], 64);
                std::memcpy(e->raw[k], st.raw[kk], sizeof e->raw[k]);
            }
            // (first person, HOOKSHOT_VR_FWD / HOOKSHOT_VR_UP, metres: the eyes are moved forward (and up) along the game's camera, to the player's head
            // behind which the game keeps its camera. The camera looks down its own -z, up is +y)
            e->E[k][11] -= fp_fwd(), e->E[k][7] += fp_up();
            eye_of_pose(cur, e->E[k], e->V[k], e->C[k], e->P[k]);
        } else
            eye_of(cur, sepx, e->V[k], e->C[k], e->P[k]);
        mul4(e->P[k], e->V[k], e->VP[k]);
        if (e->has_prev) {
            // the game blends each frame with the last one (history textures shared by both eyes): what an eye finds in the
            // history is the OTHER eye's picture. Eye 0 (drawn first) finds the last frame's eye 1; eye 1 finds this frame's eye 0.
            // alternate-eye rendering: the frame before was drawn for the OTHER eye, so that is what the game's history holds
            static const bool own_history = [] { const char* h = getenv("HOOKSHOT_VR_OWNHISTORY"); return h && *h && *h != '0'; }();
            static const bool no_history = [] { const char* h = getenv("HOOKSHOT_VR_NOHISTORY"); return h && *h && *h != '0'; }();
            // (HOOKSHOT_VR_NOHISTORY=1, a test: the last frame's camera is far to the side, so nothing of it lines up and the game throws its history away)
            if (st.pose_ok && st.last_pose && !no_history) {
                float le[16];
                std::memcpy(le, st.lastE[own_history ? k : 1 - k], 64);
                le[11] -= fp_fwd(), le[7] += fp_up();
                eye_of_pose(last, le, e->pV[k], e->pC[k], e->pP[k]);
            }
            else
                eye_of(last, no_history ? sepx + 3000.f : (own_history ? sepx : -sepx), e->pV[k], e->pC[k], e->pP[k]);
            mul4(e->pP[k], e->pV[k], e->pVP[k]);
        }
    }
    return e;
}

bool same(const float* a, const float* b, int n) {
    for (int i = 0; i < n; i++)
        if (!nearly(a[i], b[i], 1e-4f)) return false;
    return true;
}
bool same_t(const float* a, const float* x, int n16) {   // a holds x column by column
    (void)n16;
    float t[16];
    transpose4(x, t);
    return same(a, t, 16);
}
bool is_product(const float* m, const Cam& c) {   // m x (view to world) = projection: m is projection x view
    float t[16];
    mul4(m, c.C, t);
    for (int i = 0; i < 16; i++)
        if (std::fabs(t[i] - c.P[i]) > 2e-3f * (1.f + std::fabs(c.P[i]))) return false;
    return true;
}

// a perspective projection with the game's own sideways and up scales but its own depth range (a pass with a different near or far plane:
// distant trees, grass): [sx 0 x 0; 0 sy y 0; 0 0 A B; 0 0 -1 0]
bool loose_proj(const float* t, const Cam& c) {
    auto z = [](float x) { return std::fabs(x) < 2e-3f; };
    return std::fabs(t[0] - c.P[0]) < 2e-3f * std::fabs(c.P[0]) && std::fabs(t[5] - c.P[5]) < 2e-3f * std::fabs(c.P[5]) && z(t[1]) && z(t[3]) &&
           z(t[4]) && z(t[7]) && z(t[8]) && z(t[9]) && z(t[12]) && z(t[13]) && std::fabs(t[14] + 1.f) < 2e-3f && z(t[15]) && std::fabs(t[2]) < 0.1f &&
           std::fabs(t[6]) < 0.1f;
}
void loose_set(float* p, const float* Pe) {   // the eye's sideways and up scales and shear, the pass's own depth range
    p[0] = Pe[0], p[5] = Pe[5], p[2] = Pe[2], p[6] = Pe[6];
}
// m = (that kind of projection) x (the game's view): true, with the projection in pj, if so
bool loose_product(const float* m, const Cam& c, float* pj) {
    for (int j = 0; j < 4; j++) {   // (row 3 first: it is cheap, and nearly always rules it out)
        float s = 0.f;
        for (int k = 0; k < 4; k++) s += m[12 + k] * c.C[k * 4 + j];
        if (std::fabs(s - (j == 2 ? -1.f : 0.f)) > 2e-3f) return false;
    }
    mul4(m, c.C, pj);
    return loose_proj(pj, c);
}

// A view (or view-to-world) matrix that is NOT the camera this frame found but one beside it: the same place within a metre and the same way
// within about 12 degrees. The game keeps more than one copy of its camera (the last frame's, one for a pass of its own); a copy left as
// the game wrote it is drawn from the game's camera, not the eye's. The eye's offset from the game's camera (E) is applied to it as to the
// first. Rows 0-2 are the 3x4; when row 3 is the projection's x scale times row 0 (the clip x row), it is made again.
bool near_cam(float* a, bool full, const Eyes& d, int k, const Cam& c) {
    if (!d.has_pose) return false;
    float off = 0.f;
    for (int i : {1, 2, 4, 6, 8, 9}) off = std::max(off, std::fabs(a[i]));
    if (off < 0.05f) return false;
    const float* r[3] = {a, a + 4, a + 8};
    for (int i = 0; i < 3; i++) {
        const float l = r[i][0] * r[i][0] + r[i][1] * r[i][1] + r[i][2] * r[i][2];
        if (std::fabs(l - 1.f) > 3e-3f) return false;
        for (int j = i + 1; j < 3; j++)
            if (std::fabs(r[i][0] * r[j][0] + r[i][1] * r[j][1] + r[i][2] * r[j][2]) > 3e-3f) return false;
    }
    float tv = 0.f, tc = 0.f;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) tv += a[i * 4 + j] * c.V[i * 4 + j], tc += a[i * 4 + j] * c.V[j * 4 + i];
    const float pos[3] = {c.C[3], c.C[7], c.C[11]}, t[3] = {a[3], a[7], a[11]};
    auto dist = [](const float* p, const float* q) {
        return std::sqrt((p[0] - q[0]) * (p[0] - q[0]) + (p[1] - q[1]) * (p[1] - q[1]) + (p[2] - q[2]) * (p[2] - q[2]));
    };
    bool vlike = false, clike = false;
    if (tv >= 2.94f) {   // (a view: its camera position is -R^T t)
        float p[3];
        for (int j = 0; j < 3; j++) p[j] = -(a[j] * t[0] + a[4 + j] * t[1] + a[8 + j] * t[2]);
        vlike = dist(p, pos) < 1.5f;
    } else if (tc >= 2.94f)   // (a view-to-world: its position is its translation)
        clike = dist(t, pos) < 1.5f;
    if (!vlike && !clike) return false;
    const float X[16] = {a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], 0.f, 0.f, 0.f, 1.f};
    float out[16];
    const float* E = d.E[k];
    if (vlike) {
        float Ei[16];
        rigid_inv4(E, Ei);
        mul4(Ei, X, out);
    } else
        mul4(X, E, out);
    const float old0[4] = {a[0], a[1], a[2], a[3]};
    std::memcpy(a, out, 48);
    if (vlike && full) {
        bool clip = true;
        for (int j = 0; j < 4; j++) clip = clip && std::fabs(a[12 + j] - c.P[0] * old0[j]) < 2e-3f * (1.f + std::fabs(a[12 + j]));
        if (clip)
            for (int j = 0; j < 4; j++) a[12 + j] = d.P[k][0] * a[j] + d.P[k][2] * a[8 + j];
    }
    return true;
}

// camera-relative rendering: m = projection x the view's ROTATION alone (the world has been moved so the camera is at the origin)
bool rel_product(const float* m, const Cam& c) {
    float Cr[16];
    std::memcpy(Cr, c.C, 64);
    Cr[3] = Cr[7] = Cr[11] = 0.f;
    for (int j = 0; j < 4; j++) {   // (row 3 first: cheap)
        float s = 0.f;
        for (int k = 0; k < 4; k++) s += m[12 + k] * Cr[k * 4 + j];
        if (std::fabs(s - c.P[12 + j]) > 2e-3f) return false;
    }
    float t[16];
    mul4(m, Cr, t);
    for (int i = 0; i < 16; i++)
        if (std::fabs(t[i] - c.P[i]) > 2e-3f * (1.f + std::fabs(c.P[i]))) return false;
    return true;
}
void rel_make(const float* P, const float* V, float* out) {   // the eye's projection x the eye's view rotation
    float Vr[16];
    std::memcpy(Vr, V, 64);
    Vr[3] = Vr[7] = Vr[11] = 0.f;
    mul4(P, Vr, out);
}

bool rot_only(const float* a, const float* m) {   // a is m's rotation as 3 rows of 4 with a zero translation column (and m is not about the identity)
    static const int idx[9] = {0, 1, 2, 4, 5, 6, 8, 9, 10};
    float off = 0.f;
    for (int i : {1, 2, 4, 6, 8, 9}) off = std::max(off, std::fabs(m[i]));
    if (off < 0.01f) return false;
    for (int i : idx)
        if (!nearly(a[i], m[i], 1e-4f)) return false;
    return nearly(a[3], 0.f, 1e-4f) && nearly(a[7], 0.f, 1e-4f) && nearly(a[11], 0.f, 1e-4f);
}
void rot_write(float* a, const float* m) {
    static const int idx[9] = {0, 1, 2, 4, 5, 6, 8, 9, 10};
    for (int i : idx) a[i] = m[i];
}

int patch_buffer(float* f, size_t nf, const Eyes& d, int k) {
    int changed = 0;
    float t[16];
    // (tests: HOOKSHOT_VR_OFF is a list of letters naming patches to leave out: p the last frame's camera, v bare positions and directions,
    // r rotation-only matrices, l the loose projections and camera-relative products)
    static const std::string off = [] { const char* e = getenv("HOOKSHOT_VR_OFF"); return std::string(e ? e : ""); }();
    static const bool no_prev = off.find('p') != std::string::npos, no_vec = off.find('v') != std::string::npos,
                      no_rot = off.find('r') != std::string::npos, no_loose = off.find('l') != std::string::npos,
                      no_near = off.find('n') != std::string::npos;   // n: the camera copies beside the game's
    for (size_t o = 0; o + 12 <= nf; o += 4) {
        float* a = f + o;
        const bool full = o + 16 <= nf;
        for (int pass = 0; pass < 2; pass++) {   // this frame's camera, then the last frame's
            if (pass == 1 && (!d.has_prev || no_prev)) break;
            const Cam& c = pass == 0 ? d.game : d.prev;
            const float* V = pass == 0 ? d.V[k] : d.pV[k];
            const float* C = pass == 0 ? d.C[k] : d.pC[k];
            const float* P = pass == 0 ? d.P[k] : d.pP[k];
            const float* VP = pass == 0 ? d.VP[k] : d.pVP[k];
            if (same(a, c.V, 12)) { std::memcpy(a, V, 48), st.counts[0]++, changed++, o += 8; goto next; }
            if (same(a, c.C, 12)) { std::memcpy(a, C, 48), st.counts[1]++, changed++, o += 8; goto next; }
            // what some passes (foliage, billboards, sky) keep instead: the camera's rotation alone (nothing in the translation column),
            // or its position as a bare vector. Left as the game wrote them they were drawn from a different camera than the rest.
            if (pass == 0 && !no_near && near_cam(a, full, d, k, c)) { st.counts[12]++, changed++, o += 8; goto next; }
            if (!no_rot && rot_only(a, c.V)) { rot_write(a, V), st.counts[5]++, changed++, o += 8; goto next; }
            if (!no_rot && rot_only(a, c.C)) { rot_write(a, C), st.counts[6]++, changed++, o += 8; goto next; }
            if (!no_vec && std::fabs(c.C[3]) + std::fabs(c.C[7]) + std::fabs(c.C[11]) > 0.5f && nearly(a[0], c.C[3], 1e-4f) && nearly(a[1], c.C[7], 1e-4f) &&
                nearly(a[2], c.C[11], 1e-4f)) {
                a[0] = C[3], a[1] = C[7], a[2] = C[11], st.counts[7]++, changed++;
                continue;
            }
            {   // the camera's right, up and back directions as bare vectors (or their opposites): billboards face by them
                bool hit = false;
                for (int ax = 0; ax < 3 && !hit && !no_vec; ax++) {
                    const float g[3] = {c.C[ax], c.C[4 + ax], c.C[8 + ax]};
                    int big = 0;
                    for (float x : g) big += std::fabs(x) > 0.02f;
                    if (big < 2) continue;
                    for (float s : {1.f, -1.f})
                        if (nearly(a[0], s * g[0], 1e-4f) && nearly(a[1], s * g[1], 1e-4f) && nearly(a[2], s * g[2], 1e-4f)) {
                            a[0] = s * C[ax], a[1] = s * C[4 + ax], a[2] = s * C[8 + ax];
                            st.counts[11]++, changed++, hit = true;
                            break;
                        }
                }
                if (hit) continue;
            }
            if (!full) continue;
            if (same(a, c.P, 16)) { std::memcpy(a, P, 64), st.counts[2]++, changed++, o += 12; goto next; }
            if (same_t(a, c.V, 16)) { transpose4(V, t), std::memcpy(a, t, 64), st.counts[4]++, changed++, o += 12; goto next; }
            if (same_t(a, c.C, 16)) { transpose4(C, t), std::memcpy(a, t, 64), st.counts[4]++, changed++, o += 12; goto next; }
            if (same_t(a, c.P, 16)) { transpose4(P, t), std::memcpy(a, t, 64), st.counts[4]++, changed++, o += 12; goto next; }
            if (is_product(a, c)) { std::memcpy(a, VP, 64), st.counts[3]++, changed++, o += 12; goto next; }
            transpose4(a, t);
            if (is_product(t, c)) { transpose4(VP, t), std::memcpy(a, t, 64), st.counts[3]++, changed++, o += 12; goto next; }
            if (!no_loose) {   // the same kinds with the pass's own depth range
                float pj[16], w[16];
                if (loose_proj(a, c)) { loose_set(a, P), st.counts[8]++, changed++, o += 12; goto next; }
                if (loose_proj(t, c)) { loose_set(t, P), transpose4(t, a), st.counts[8]++, changed++, o += 12; goto next; }
                if (loose_product(a, c, pj)) { loose_set(pj, P), mul4(pj, V, w), std::memcpy(a, w, 64), st.counts[9]++, changed++, o += 12; goto next; }
                if (loose_product(t, c, pj)) { loose_set(pj, P), mul4(pj, V, w), transpose4(w, a), st.counts[9]++, changed++, o += 12; goto next; }
                if (rel_product(a, c)) { rel_make(P, V, w), std::memcpy(a, w, 64), st.counts[10]++, changed++, o += 12; goto next; }
                if (rel_product(t, c)) { rel_make(P, V, w), transpose4(w, a), st.counts[10]++, changed++, o += 12; goto next; }
            }
        }
    next:;
    }
    // (test: HOOKSHOT_VR_SCALARS=1) a buffer that held a camera may also hold the game's field of view as lone numbers (the projection's x and y
    // scale, or their inverses), for a pass that places things itself. They are made the eye's.
    // (HOOKSHOT_VR_SCALARS: 1 all four, p the scales only, i their inverses only)
    static const std::string scal_mode = [] { const char* e = getenv("HOOKSHOT_VR_SCALARS"); return std::string(e ? e : ""); }();
    static const bool scalars = !scal_mode.empty() && scal_mode != "0";
    static const bool sc_p = scal_mode.find('1') != std::string::npos || scal_mode.find('p') != std::string::npos,
                      sc_i = scal_mode.find('1') != std::string::npos || scal_mode.find('i') != std::string::npos;
    if (scalars && changed > 0 && std::fabs(d.game.P[0]) > 1e-3f && std::fabs(d.game.P[5]) > 1e-3f && std::fabs(d.P[k][0]) > 1e-3f && std::fabs(d.P[k][5]) > 1e-3f) {
        const float src[4] = {d.game.P[0], d.game.P[5], 1.f / d.game.P[0], 1.f / d.game.P[5]};
        const float dst[4] = {d.P[k][0], d.P[k][5], 1.f / d.P[k][0], 1.f / d.P[k][5]};
        for (size_t i = 0; i < nf; i++)
            for (int j = 0; j < 4; j++)
                if ((j < 2 ? sc_p : sc_i) && f[i] != dst[j] && std::fabs(f[i] - src[j]) <= 1e-5f * std::fabs(src[j])) {
                    f[i] = dst[j];
                    st.counts[13]++;
                    break;
                }
    }
    return changed;
}

bool make_sc(uint32_t w, uint32_t h, XrSwapchain& sc, std::vector<VkImage>& images) {
    XrSwapchainCreateInfo ci{XR_TYPE_SWAPCHAIN_CREATE_INFO};
    ci.usageFlags = XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT | XR_SWAPCHAIN_USAGE_TRANSFER_DST_BIT | XR_SWAPCHAIN_USAGE_SAMPLED_BIT;
    ci.format = g.format;
    ci.sampleCount = 1;
    ci.width = w;
    ci.height = h;
    ci.faceCount = 1;
    ci.arraySize = 1;
    ci.mipCount = 1;
    {
        std::lock_guard qk(dev.present_mu);
        if (!XR_CHK(xrCreateSwapchain(g.session, &ci, &sc))) return false;
    }
    uint32_t n = 0;
    if (!XR_CHK(xrEnumerateSwapchainImages(sc, 0, &n, nullptr))) return false;
    std::vector<XrSwapchainImageVulkan2KHR> imgs(n, XrSwapchainImageVulkan2KHR{XR_TYPE_SWAPCHAIN_IMAGE_VULKAN2_KHR});
    if (!XR_CHK(xrEnumerateSwapchainImages(sc, n, &n, (XrSwapchainImageBaseHeader*)imgs.data()))) return false;
    images.clear();
    for (auto& i : imgs) images.push_back(i.image);
    return true;
}
void destroy_eyes() {
    if (!st.sc[0] && !st.sc[1]) return;
    wait_slots();
    std::lock_guard qk(dev.present_mu);
    for (int e = 0; e < 2; e++) {
        if (st.sc[e]) xrDestroySwapchain(st.sc[e]);
        st.sc[e] = XR_NULL_HANDLE;
        st.images[e].clear();
    }
}
void destroy_cache() {
    for (int e = 0; e < 2; e++) {
        if (st.cimg[e]) vkDestroyImage(dev.device, st.cimg[e], nullptr);
        if (st.cmem[e]) vkFreeMemory(dev.device, st.cmem[e], nullptr);
        st.cimg[e] = VK_NULL_HANDLE, st.cmem[e] = VK_NULL_HANDLE, st.cvalid[e] = false;
    }
    st.cw = st.ch = 0;
}
bool ensure_cache(uint32_t w, uint32_t h) {
    if (st.cimg[0] && st.cimg[1] && st.cw == w && st.ch == h) return true;
    wait_slots();
    destroy_cache();
    for (int e = 0; e < 2; e++) {
        VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ci.imageType = VK_IMAGE_TYPE_2D;
        ci.format = (VkFormat)g.format;
        ci.extent = {w, h, 1};
        ci.mipLevels = 1;
        ci.arrayLayers = 1;
        ci.samples = VK_SAMPLE_COUNT_1_BIT;
        ci.tiling = VK_IMAGE_TILING_OPTIMAL;
        ci.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        if (vkCreateImage(dev.device, &ci, nullptr, &st.cimg[e]) != VK_SUCCESS) { destroy_cache(); return false; }
        VkMemoryRequirements mr;
        vkGetImageMemoryRequirements(dev.device, st.cimg[e], &mr);
        uint32_t type = UINT32_MAX;
        for (uint32_t i = 0; i < dev.memory.memoryTypeCount; i++)
            if ((mr.memoryTypeBits & (1u << i)) && (dev.memory.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
                type = i;
                break;
            }
        if (type == UINT32_MAX) { destroy_cache(); return false; }
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = mr.size;
        ai.memoryTypeIndex = type;
        if (vkAllocateMemory(dev.device, &ai, nullptr, &st.cmem[e]) != VK_SUCCESS ||
            vkBindImageMemory(dev.device, st.cimg[e], st.cmem[e], 0) != VK_SUCCESS) { destroy_cache(); return false; }
    }
    st.cw = w, st.ch = h;
    mkwvk::logf("vr: stereo picture cache %ux%u made", w, h);
    return true;
}
bool ensure_eyes(uint32_t w, uint32_t h) {
    if (st.sc[0] && st.sc[1] && st.w == w && st.h == h) return true;
    destroy_eyes();
    if (!pick_format()) return false;
    for (int e = 0; e < 2; e++)
        if (!make_sc(w, h, st.sc[e], st.images[e])) return false;
    st.w = w, st.h = h;
    mkwvk::logf("vr: stereo screens %ux%u, 2 x %zu images, format %lld", w, h, st.images[0].size(), (long long)g.format);
    return true;
}

} // namespace


// ---- where the game keeps its camera (hookshot_vr_scan.txt) ----------------------------------------------------------------
// The camera's position is known from the uniform buffer. The game's own memory is searched for three floats that match it:
// the struct that holds them is what the game's culling and its camera code use. What lies around each hit goes to the log.
#ifdef _WIN32
void scan_guest_memory(float px, float py, float pz) {
    mkwvk::logf("vr-scan: looking for the camera position (%.2f, %.2f, %.2f) in the game's memory", px, py, pz);
    const float tol = 0.05f;
    const size_t chunk = 1u << 22, overlap = 256;
    std::vector<uint8_t> buf(chunk + overlap);
    int hits = 0;
    uint64_t scanned = 0;
    const HANDLE self = GetCurrentProcess();
    MEMORY_BASIC_INFORMATION mbi;
    uint8_t* a = nullptr;
    while (VirtualQuery(a, &mbi, sizeof mbi) == sizeof mbi && hits < 80) {
        int region_hits = 0;
        const uint8_t* base = static_cast<const uint8_t*>(mbi.BaseAddress);
        const bool ok = mbi.State == MEM_COMMIT && (mbi.Protect & (PAGE_READWRITE | PAGE_EXECUTE_READWRITE | PAGE_WRITECOPY)) &&
                        !(mbi.Protect & (PAGE_GUARD | PAGE_NOACCESS));
        if (ok) {
            for (size_t off = 0; off < mbi.RegionSize && hits < 80 && region_hits < 6; off += chunk) {
                const size_t want = std::min(chunk + overlap, (size_t)mbi.RegionSize - off);
                SIZE_T got = 0;
                if (!ReadProcessMemory(self, base + off, buf.data(), want, &got) || got < 16) continue;
                scanned += got;
                const float* f = reinterpret_cast<const float*>(buf.data());
                const size_t nf = got / 4;
                if ((off / chunk) % 8 == 7) Sleep(1);
                for (size_t i = 0; i + 3 <= nf && i * 4 < chunk && hits < 80 && region_hits < 6; i++) {
                    if (!(std::fabs(f[i] - px) <= tol && std::fabs(f[i + 1] - py) <= tol && std::fabs(f[i + 2] - pz) <= tol)) continue;   // (NaN is no match)
                    ++hits, ++region_hits;
                    const uint8_t* at = base + off + i * 4;
                    char line[2048];
                    int n = snprintf(line, sizeof line, "vr-scan: hit %d at %p (region %p +%zx, type %x, prot %x): floats from -24 to +40:", hits,
                                     (const void*)at, (const void*)base, (size_t)mbi.RegionSize, (unsigned)mbi.Type, (unsigned)mbi.Protect);
                    float ctx[65];
                    SIZE_T cg = 0;
                    bool have = ReadProcessMemory(self, at - 96, ctx, sizeof ctx, &cg) && cg == sizeof ctx;
                    for (int k = 0; have && k < 64 && n < (int)sizeof line - 24; k++) n += snprintf(line + n, sizeof line - n, " %g", ctx[k]);
                    mkwvk::logf("%s", line);
                }
            }
        }
        a = const_cast<uint8_t*>(base) + mbi.RegionSize;
    }
    mkwvk::logf("vr-scan: done, %d hit(s) in %.0f MB", hits, scanned / 1048576.0);
}
#endif

// ---- the game's field of view (hookshot_vr_fov.txt) -------------------------------------------------------------------------
// The game's perspective projection object holds near, far, the vertical field of view in radians, its sin, cos and tan, then the
// aspect ratio (sead's PerspectiveProjection; the matrix update and the culling both read the tan and the aspect). Wherever such an
// object is found, those four are kept at the field of view in the file (radians; the file's presence turns it on). The game
// writes them again from its camera every frame: this thread writes them again over those, a thousand times a second.
#ifdef _WIN32
namespace {
int fov_object(const float* f) {   // f[0] = fov, f[1] sin, f[2] cos, f[3] tan, f[4] aspect, f[-2] near, f[-1] far. 1: the angle itself, 2: half of it
    if (!(f[0] > 0.2f && f[0] < 2.6f)) return 0;
    if (!(f[4] > 1.0f && f[4] < 4.2f && f[-2] > 0.f && f[-2] < 10.f && f[-1] > 100.f)) return 0;
    auto close = [](float x, float y) { return std::fabs(x - y) < 3e-4f; };
    if (close(f[1], std::sin(f[0] * 0.5f)) && close(f[2], std::cos(f[0] * 0.5f)) && close(f[3], std::tan(f[0] * 0.5f))) return 2;
    if (close(f[1], std::sin(f[0])) && close(f[2], std::cos(f[0])) && close(f[3], std::tan(f[0]))) return 1;
    return 0;
}
std::vector<std::pair<uint8_t*, int>> find_fov_objects() {
    std::vector<std::pair<uint8_t*, int>> out;
    const size_t chunk = 1u << 22, overlap = 256;
    std::vector<uint8_t> buf(chunk + overlap);
    const HANDLE self = GetCurrentProcess();
    MEMORY_BASIC_INFORMATION mbi;
    uint8_t* a = nullptr;
    size_t nreg = 0, mb = 0;
    const auto t0 = std::chrono::steady_clock::now();
    while (VirtualQuery(a, &mbi, sizeof mbi) == sizeof mbi && out.size() < 64) {
        const uint8_t* base = static_cast<const uint8_t*>(mbi.BaseAddress);
        const bool ok = mbi.State == MEM_COMMIT && mbi.Type == MEM_PRIVATE && (mbi.Protect & (PAGE_READWRITE | PAGE_EXECUTE_READWRITE)) &&
                        !(mbi.Protect & (PAGE_GUARD | PAGE_NOACCESS));
        if (ok) {
            ++nreg;
            mb += mbi.RegionSize >> 20;
            if ((mbi.RegionSize >> 20) >= 512)
                mkwvk::logf("vr-fov: searching a %zu MB region (%zu MB so far)", (size_t)(mbi.RegionSize >> 20), mb);
        }
        if (ok)
            for (size_t off = 0; off < mbi.RegionSize && out.size() < 64; off += chunk) {
                const size_t want = std::min(chunk + overlap, (size_t)mbi.RegionSize - off);
                SIZE_T got = 0;
                if (!ReadProcessMemory(self, base + off, buf.data(), want, &got) || got < 64) continue;
                const float* f = reinterpret_cast<const float*>(buf.data());
                const size_t nf = got / 4;
                for (size_t i = 2; i + 5 <= nf && i * 4 < chunk && out.size() < 64; i++)
                    if (const int kind = (f[i - 1] >= 10000.f && f[i - 2] <= 0.5f) ? fov_object(f + i) : 0) {   // (the main camera's: far plane 25000)
                        out.push_back({const_cast<uint8_t*>(base) + off + i * 4, kind});
                        mkwvk::logf("vr-fov: object at %p: fov %.4f, aspect %.4f, near %g, far %g, %s angle", (void*)(base + off + i * 4), (double)f[i], (double)f[i + 4], (double)f[i - 2], (double)f[i - 1], kind == 2 ? "half" : "full");
                    }
            }
        a = const_cast<uint8_t*>(base) + mbi.RegionSize;
        if (ok) Sleep(1);   // (the search must not stall the game)
    }
    mkwvk::logf("vr-fov: searched %zu regions, %zu MB, %lld ms", nreg, mb,
                (long long)std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count());
    return out;
}
float fov_file_value() {
    FILE* f = fopen("hookshot_vr_fov.txt", "rb");
    if (!f) return 0.f;
    char b[32] = {};
    const size_t n = fread(b, 1, sizeof b - 1, f);
    fclose(f);
    b[n] = 0;
    const float v = (float)atof(b);
    return v > 0.3f && v < 2.6f ? v : 0.f;
}
std::mutex fov_mu;
std::vector<std::pair<uint8_t*, int>> fov_objs;
std::atomic<bool> fov_scanning{false};
void fov_scan_thread() {
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
    mkwvk::logf("vr-fov: searching (in the background; it takes a few minutes)");
    auto found = find_fov_objects();
    {
        std::lock_guard lk(fov_mu);
        fov_objs = found;
    }
    mkwvk::logf("vr-fov: %zu main-camera projection object(s) found", found.size());
    fov_scanning = false;
}
void fov_keeper() {
    mkwvk::logf("vr-fov: keeper started (hookshot_vr_fov.txt holds the vertical field of view in radians; delete the file to stop)");
    auto last_scan = std::chrono::steady_clock::now() - std::chrono::seconds(100);
    const HANDLE self = GetCurrentProcess();
    std::vector<std::pair<uint8_t*, int>> objs;
    int bad_rounds = 0;
    for (;;) {
        const float v = fov_file_value();
        if (v <= 0.f) {
            Sleep(200);
            continue;
        }
        const auto now = std::chrono::steady_clock::now();
        {
            std::lock_guard lk(fov_mu);
            objs = fov_objs;
        }
        if (!fov_scanning.load() && (objs.empty() || bad_rounds > 3000) && now - last_scan > std::chrono::seconds(20)) {
            fov_scanning = true;
            last_scan = now;
            bad_rounds = 0;
            std::thread(fov_scan_thread).detach();
        }
        size_t bad = 0;
        for (auto& ob : objs) {
            uint8_t* o = ob.first;
            const float k = ob.second == 2 ? 0.5f : 1.0f;
            const float w[4] = {v, std::sin(v * k), std::cos(v * k), std::tan(v * k)};
            float cur[7];   // (two floats before the object: near and far)
            SIZE_T r = 0;
            if (!ReadProcessMemory(self, o - 8, cur, sizeof cur, &r) || r != sizeof cur || !fov_object(cur + 2)) {   // (freed or reused: left alone)
                ++bad;
                continue;
            }
            SIZE_T x = 0;
            WriteProcessMemory(self, o, w, sizeof w, &x);
        }
        if (!objs.empty() && bad * 2 >= objs.size()) ++bad_rounds;   // (most of them gone for a few seconds: a new scene. Search again)
        else bad_rounds = 0;
        Sleep(1);
    }
}
}
#endif

void stereo_init() {
    {
        const char* ce = getenv("HOOKSHOT_VR_CAMCTL");
        cc.enabled = ce && *ce && *ce != '0';
        if (const char* l = getenv("HOOKSHOT_VR_CAMLAG")) cc.lag = std::max(0, std::min(8, atoi(l)));
        if (const char* d = getenv("HOOKSHOT_VR_CAMDIST")) cc.dist = (float)atof(d);
        if (const char* y = getenv("HOOKSHOT_VR_CAMYAW")) cc.ysign = atof(y) < 0 ? -1.f : 1.f;
        if (const char* p = getenv("HOOKSHOT_VR_CAMPITCH")) cc.psign = atof(p) < 0 ? -1.f : 1.f;
        if (cc.enabled) mkwvk::logf("vr-cam: the game's camera follows the head (HOOKSHOT_VR_CAMCTL), lag %d, distance %.2f, yaw sign %.0f, pitch sign %.0f", cc.lag, (double)cc.dist, (double)cc.ysign, (double)cc.psign);
    }
    const char* e = getenv("HOOKSHOT_VR_STEREO");
    st.on = e && *e && *e != '0';
    st.conv = env_float("HOOKSHOT_VR_CONVERGENCE", 190.f);
    st.sep_fixed = getenv("HOOKSHOT_VR_SEPARATION") && *getenv("HOOKSHOT_VR_SEPARATION");
    st.sep = st.sep_fixed ? env_float("HOOKSHOT_VR_SEPARATION", 6.4f) : st.ipd * st.conv / g.distance;
    if (st.on)
        mkwvk::logf("vr: stereo is on (eyes %.2f game units apart, meeting at %.0f; HOOKSHOT_VR_SEPARATION / _CONVERGENCE change them)",
                    st.sep, st.conv);
}
bool stereo_take_handled() {
    const bool h = st.handled;
    st.handled = false;
    return h;
}
bool stereo_active() { return st.on && st.go && g.created && g.running && !g.dead; }
bool stereo_ready() { return stereo_active() && st.have.load() && st.patched.load(); }
int stereo_mode() {
    if (!st.on || !g.created || g.dead) return 0;
    if (!st.have.load()) return 1;
    return st.eye.load() >= 0 ? 2 : 0;
}

void update_poses() {   // (each game frame: where the headset's eyes will be when this frame is shown)
    if (!head_tracking_enabled() || !xrLocateViews || !st.time || !g.space) return;
    XrViewLocateInfo li{XR_TYPE_VIEW_LOCATE_INFO};
    li.viewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
    li.displayTime = st.time + (XrTime)st.period;
    li.space = g.space;
    XrViewState vs{XR_TYPE_VIEW_STATE};
    XrView v[2] = {{XR_TYPE_VIEW}, {XR_TYPE_VIEW}};
    uint32_t n = 0;
    if (!XR_SUCCEEDED(xrLocateViews(g.session, &li, &vs, 2, &n, v)) || n != 2) return;
    const XrViewStateFlags need = XR_VIEW_STATE_POSITION_VALID_BIT | XR_VIEW_STATE_ORIENTATION_VALID_BIT;
    if ((vs.viewStateFlags & need) != need) return;
    std::lock_guard lk(st.mu);
    for (int e = 0; e < 2; e++) {
        const XrPosef& p = v[e].pose;
        st.raw[e][0] = p.position.x, st.raw[e][1] = p.position.y, st.raw[e][2] = p.position.z;
        st.raw[e][3] = p.orientation.x, st.raw[e][4] = p.orientation.y, st.raw[e][5] = p.orientation.z, st.raw[e][6] = p.orientation.w;
    }
    float M0[16], M1[16];
    pose_to_mat4(st.raw[0], M0);
    pose_to_mat4(st.raw[1], M1);
    if (st.recenter) {   // (straight ahead is where the head faces now; the game's camera is there)
        const float cx = 0.5f * (st.raw[0][0] + st.raw[1][0]), cy = 0.5f * (st.raw[0][1] + st.raw[1][1]), cz = 0.5f * (st.raw[0][2] + st.raw[1][2]);
        const float fx = -M0[2], fz = -M0[10];   // (the head's forward: its -z axis)
        const float th = std::atan2(-fx, -fz);
        const float cs = std::cos(th), sn = std::sin(th);
        float A[16] = {cs, 0, sn, cx, 0, 1, 0, cy, -sn, 0, cs, cz, 0, 0, 0, 1};
        std::memcpy(st.A, A, sizeof A);
        st.yaw0 = th;
        rigid_inv4(A, st.A_inv);
        st.recenter = false;
        mkwvk::logf("vr: head tracking: straight ahead set (head at %.2f %.2f %.2f, turned %.0f degrees)", cx, cy, cz, th * 57.2958f);
    }
    mul4(st.A_inv, M0, st.Erel[0]);
    mul4(st.A_inv, M1, st.Erel[1]);
    static const bool rotate = [] { const char* e = getenv("HOOKSHOT_VR_ROTATE"); return !(e && *e == '0'); }();
    if (!rotate) {   // (a test: the eyes move with the head, but the picture is not turned: a window that stays where it is)
        for (int e = 0; e < 2; e++) {
            const float t[3] = {st.Erel[e][3], st.Erel[e][7], st.Erel[e][11]};
            float M[16] = {1, 0, 0, t[0], 0, 1, 0, t[1], 0, 0, 1, t[2], 0, 0, 0, 1};
            std::memcpy(st.Erel[e], M, sizeof M);
            float w[16];
            mul4(st.A, M, w);
            st.raw[e][0] = w[3], st.raw[e][1] = w[7], st.raw[e][2] = w[11];
            st.raw[e][3] = 0.f, st.raw[e][4] = std::sin(st.yaw0 * 0.5f), st.raw[e][5] = 0.f, st.raw[e][6] = std::cos(st.yaw0 * 0.5f);
        }
    }
    if (cc.enabled) {
        cam_service();
        CamHist& h = cc.hist[st.tick & 15];
        h.tick = st.tick;
        std::memcpy(h.E, st.Erel, sizeof h.E);
        std::memcpy(h.raw, st.raw, sizeof h.raw);
        const float* r = st.Erel[0];   // (the head's rotation: yaw, pitch, roll as Ry * Rx * Rz)
        const double ph = std::asin(std::max(-1.f, std::min(1.f, -r[6]))), th = std::atan2((double)r[2], (double)r[10]);
        h.driven = cam_live() > 0;
        if (h.driven && !cc.ctl.empty() && cam_valid(cc.ctl[0])) {
            const float d = cam_getf(cc.ctl[0] + 0x10);
            if (d > 0.3f && d < 60.f) cc.live_dist = d;
        }
        if (h.driven) {
            if (!cc.have_last) cc.pushed_th = th, cc.pushed_ph = ph, cc.have_last = true;
            double dth = th - cc.pushed_th;
            if (dth > 3.14159265) dth -= 6.2831853;
            if (dth < -3.14159265) dth += 6.2831853;
            double dph = ph - cc.pushed_ph;
            if (std::fabs(dth) < 0.0003) dth = 0;   // (a headset's tracking noise is not pushed into the game: it would walk the camera off)
            if (std::fabs(dph) < 0.0003) dph = 0;
            if (dth != 0 || dph != 0) {
                cam_push(dth, dph);
                cc.pushed_th += dth, cc.pushed_ph += dph;
                if (cc.pushed_th > 3.14159265) cc.pushed_th -= 6.2831853;
                if (cc.pushed_th < -3.14159265) cc.pushed_th += 6.2831853;
            } else
                cam_hold_pitch();
            if ((++cc.pushes % 600) == 1)
                mkwvk::logf("vr-cam: head yaw %.1f pitch %.1f; given to the game %.1f %.1f; the game's yaw %.1f pitch %.1f distance %.2f", th * 57.29578,
                            ph * 57.29578, cc.pushed_th * 57.29578, cc.pushed_ph * 57.29578, (double)cam_getf(cc.ctl[0] + 0x2c),
                            (double)cam_getf(cc.ctl[0] + 0x14), (double)cam_getf(cc.ctl[0] + 0x10));
        } else
            cc.have_last = false, cc.have_pcmd = false;
        // what the game's camera has been turned by: the picture draws only what is left of the head's pose
        const double uth = h.driven ? cc.pushed_th : th, uph = h.driven ? cc.pushed_ph : ph;
        const float cth = (float)std::cos(uth), sth = (float)std::sin(uth), cph = (float)std::cos(uph), sph = (float)std::sin(uph);
        const float Rp[16] = {cth, sth * sph, sth * cph, 0, 0, cph, -sph, 0, -sth, cth * sph, cth * cph, 0, 0, 0, 0, 1};
        std::memcpy(h.Rp, Rp, sizeof Rp);
    }
    st.pose_ok = true;
    if ((++st.pose_log % 600) == 1)
        mkwvk::logf("vr: head tracking: eyes at (%.3f %.3f %.3f) and (%.3f %.3f %.3f) from straight ahead", st.Erel[0][3], st.Erel[0][7], st.Erel[0][11],
                    st.Erel[1][3], st.Erel[1][7], st.Erel[1][11]);
}

void stereo_frame_start() {
    if (!st.on || !g.created) return;
    const auto now = std::chrono::steady_clock::now();
    if ((st.tick % 30) == 0) {   // (the arm file: create it to turn stereo on, delete it to turn it off, while the game runs)
        FILE* f = fopen("hookshot_vr_stereo_arm.txt", "rb");
        const bool a = f != nullptr;
        if (f) fclose(f);
        if (a != st.armed) {
            st.armed = a;
            mkwvk::logf("vr: stereo %s (file hookshot_vr_stereo_arm.txt %s)", a ? "ARMED" : "disarmed", a ? "found" : "gone");
            if (a) st.recenter = true;
        }
    }
    if ((st.tick % 30) == 19) {   // (hookshot_vr_recenter.txt: straight ahead is where the head faces now)
        if (FILE* f = fopen("hookshot_vr_recenter.txt", "rb")) {
            fclose(f);
            remove("hookshot_vr_recenter.txt");
            st.recenter = true;
        }
    }
    if ((st.tick % 30) == 11) {
        if (FILE* f = fopen("hookshot_vr_scan.txt", "rb")) {
            fclose(f);
            remove("hookshot_vr_scan.txt");
            st.scan_req = true;
        }
    }
    if ((st.tick % 30) == 17) {
        if (FILE* f = fopen("hookshot_vr_dump.txt", "rb")) {
            fclose(f);
            remove("hookshot_vr_dump.txt");
            st.dump_req = true;
        }
    }
    if (!st.fov_started && (st.tick % 30) == 13) {
        if (FILE* f = fopen("hookshot_vr_fov.txt", "rb")) {
            fclose(f);
            st.fov_started = true;
#ifdef _WIN32
            std::thread(fov_keeper).detach();
#endif
        }
    }
    hide_poll(st.tick);
    if ((st.tick % 30) == 7) {   // (which of the second eye's compute passes to leave out: numbers and ranges, "all", or nothing)
        std::vector<std::pair<int, int>> r;
        bool all = false;
        if (FILE* f = fopen("hookshot_vr_skip.txt", "rb")) {
            char buf[1024] = {};
            const size_t n = fread(buf, 1, sizeof buf - 1, f);
            fclose(f);
            buf[n] = 0;
            for (char* p = strtok(buf, " ,;\r\n\t"); p; p = strtok(nullptr, " ,;\r\n\t")) {
                if (!strcmp(p, "all")) all = true;
                else if (*p >= '0' && *p <= '9') {
                    int a = atoi(p), b = a;
                    if (const char* d = strchr(p, '-')) b = atoi(d + 1);
                    r.push_back({a, b});
                }
            }
        }
        std::lock_guard lk(st.smu);
        if (all != st.skip_all || r != st.skips) {
            st.skip_all = all, st.skips = r;
            mkwvk::logf("vr: skip list changed: %s%zu range(s)", all ? "all, " : "", r.size());
        }
    }
    if (st.last_start.time_since_epoch().count()) {
        const double dt = std::chrono::duration<double, std::milli>(now - st.last_start).count();
        if (dt > 200.0 && st.go) {   // (a long gap between frames: loading. Stereo waits until things have been calm a while)
            st.hold_until = st.tick + 300;
            mkwvk::logf("vr: stereo paused: %.0f ms between frames (loading?)", dt);
        } else if (dt > 200.0) st.hold_until = st.tick + 300;
    }
    st.last_start = now;
    st.tick++;
    const bool was = st.go;
    st.go = st.armed && st.tick >= st.hold_until && g.running && !g.dead;
    if (was && !st.go) {
        std::lock_guard lk(st.mu);
        st.eyes.reset(); st.have = false; st.eye = -1; st.left_done = false;
    }
    if (!stereo_active()) return;
    update_poses();
    std::lock_guard lk(st.mu);
    st.aer_eye ^= 1;   // (alternate-eye rendering: this game frame is for the other eye than the last)
    st.patched = false;
    st.eyes.reset();
    st.have = false;
    st.eye = -1;
    st.left_done = false;
}
void stereo_cost(double ms) {
    if (ms > 60.0) ++st.slow; else st.slow = 0;
    if (st.slow >= 8) {
        st.slow = 0;
        st.hold_until = st.tick + 600;
        mkwvk::logf("vr: stereo paused: the two eyes took %.0f ms; trying again in a while", ms);
    }
}
void stereo_sync() {
    if (!g.created || !dev.device) return;
    std::lock_guard qk(dev.present_mu);
    queue_flush_locked();
    vkQueueWaitIdle(dev.queue);
}
void stereo_frame_done() {
    std::lock_guard lk(st.mu);
    if (st.eyes) {
        st.last = st.eyes->game;
        st.last_pose = st.eyes->has_pose;
        if (st.last_pose) std::memcpy(st.lastE, st.eyes->E, sizeof st.lastE);
    }
#ifdef _WIN32
    if (st.scan_req.exchange(false) && st.last.valid) {
        const float px = st.last.C[3], py = st.last.C[7], pz = st.last.C[11];
        std::thread([px, py, pz] { scan_guest_memory(px, py, pz); }).detach();
    }
#endif
    st.eyes.reset();
    st.have = false;
    st.eye = -1;
    st.left_done = false;
}
void stereo_set_eye(int eye) {
    st.eye = eye;
    if (eye == 1) st.cidx = 0;
}
bool skip_compute(uint64_t prog, uint32_t x, uint32_t y, uint32_t z) {
    static const bool enabled = [] { const char* e = getenv("HOOKSHOT_VR_SKIPCOMPUTE"); return e && *e == '1'; }();
    if (!enabled || !st.on || st.eye.load() != 1) return false;
    const int idx = st.cidx++;
    if (st.clog < 400) {
        st.clog++;
        mkwvk::logf("vr: eye 1 compute #%d prog %016llx grid %u %u %u", idx, (unsigned long long)prog, x, y, z);
    }
    std::lock_guard lk(st.smu);
    if (st.skip_all) return true;
    for (const auto& r : st.skips)
        if (idx >= r.first && idx <= r.second) return true;
    return false;
}
bool hide_draw(uint64_t prog, bool indirect, int64_t count, int instances) {
    (void)count;
    if (!st.on || !st.armed) return false;
    std::lock_guard lk(hide_mu);
    HideInfo& h = hide_seen[prog];
    h.n++;
    if (indirect) h.ind++;
    if (instances > 1) h.inst++;
    if (!hide_any.load()) return false;
    return (hide_ind && indirect) || (hide_inst && instances > 1) || hide_keys.count(prog) != 0;
}
namespace {
std::atomic<int64_t> diag_total{0};
}
bool diag_on() { return st.on && st.armed && st.eye.load() >= 0; }
bool diag_scan_wanted(uint64_t prog) {
    return diag_on() && st.have.load() && (prog == 0xd8e59341cf4f629cull || prog == 0xd905f52b7517c681ull || prog == 0xe28434e1afe8321aull);
}
namespace {
// (diagnostics) camera-like data in any of the usual arrangements, signs and transposes included
bool diag_proj_like(const float* t, const Cam& c) {   // [sx 0 x 0; 0 +-sy y 0; 0 0 A B; 0 0 +-1 0]
    auto z = [](float x) { return std::fabs(x) < 2e-3f; };
    if (!(z(t[1]) && z(t[3]) && z(t[4]) && z(t[7]) && z(t[8]) && z(t[9]) && z(t[12]) && z(t[13]) && z(t[15]))) return false;
    if (std::fabs(std::fabs(t[14]) - 1.f) > 2e-3f) return false;
    return std::fabs(std::fabs(t[0]) - std::fabs(c.P[0])) < 3e-3f * std::fabs(c.P[0]) &&
           std::fabs(std::fabs(t[5]) - std::fabs(c.P[5])) < 3e-3f * std::fabs(c.P[5]);
}
bool diag_rot_like(const float* a) {   // three rows of four whose first three are an orthonormal frame that is not the identity
    const float* r[3] = {a, a + 4, a + 8};
    float off = 0.f;
    for (int i : {1, 2, 4, 6, 8, 9}) off = std::max(off, std::fabs(a[i]));
    if (off < 0.05f) return false;
    for (int i = 0; i < 3; i++) {
        const float l = r[i][0] * r[i][0] + r[i][1] * r[i][1] + r[i][2] * r[i][2];
        if (std::fabs(l - 1.f) > 3e-3f) return false;
        for (int j = i + 1; j < 3; j++)
            if (std::fabs(r[i][0] * r[j][0] + r[i][1] * r[j][1] + r[i][2] * r[j][2]) > 3e-3f) return false;
    }
    return true;
}
// 0: nothing; 1 a projection; 2 a view-projection; 3 a transposed one; 4 a rotation
int diag_kind(const float* w, const Cam& c) {
    if (diag_proj_like(w, c)) return 1;
    float t[16], tw[16];
    mul4(w, c.C, t);
    if (diag_proj_like(t, c)) return 2;
    transpose4(w, tw);
    mul4(tw, c.C, t);
    if (diag_proj_like(t, c)) return 3;
    if (diag_rot_like(w)) return 4;
    return 0;
}
}
int diag_scan(const uint8_t* data, uint64_t n) {   // how many camera-like windows these bytes hold
    std::shared_ptr<const Eyes> d;
    {
        std::lock_guard lk(st.mu);
        d = st.eyes;
    }
    if (!d) return 0;
    const float* f = reinterpret_cast<const float*>(data);
    const size_t nf = (size_t)(n / 4);
    int hits = 0;
    for (size_t o = 0; o + 16 <= nf; o += 4)
        if (diag_kind(f + o, d->game)) hits++;
    return hits;
}
void diag_ubo(uint64_t prog, int stage, uint32_t index, const uint8_t* data, uint64_t n, bool compute, int changed) {   // a uniform buffer's camera-like windows, once per buffer slot (compute: a compute pass's, only what is found)
    static std::mutex mu;
    static std::set<uint64_t> seen;
    static std::atomic<int> compute_lines{0};   // (a compute pass's lines are limited: only what is found is told)
    {
        std::lock_guard lk(mu);
        if (!seen.insert(prog ^ ((uint64_t)stage << 58) ^ ((uint64_t)index << 50) ^ (compute ? 0x5a5a5a5a5a5a5a5aull : 0)).second) return;
    }
    auto may_tell = [&] { return !compute || compute_lines++ < 250; };
    const char* tag = compute ? "vr-diag-cubo: COMPUTE" : "vr-diag-ubo:";
    std::shared_ptr<const Eyes> d;
    {
        std::lock_guard lk(st.mu);
        d = st.eyes;
    }
    if (!d) return;
    const float* f = reinterpret_cast<const float*>(data);
    const size_t nf = (size_t)(n / 4);
    if (compute && may_tell()) {   // (this frame's camera beside what the pass was given: why was it not changed?)
        const Cam& g0 = d->game;
        const Cam& g1 = d->prev;
        mkwvk::logf("vr-diag-cubo: COMPUTE prog %016llx uniform buffer %u (%llu bytes): %d matrices changed in it; the frame's camera: view row 0 %.4f %.4f %.4f, position %.2f %.2f %.2f, projection %.4f; last frame's (%s): row 0 %.4f %.4f %.4f, position %.2f %.2f %.2f; eye 0's projection %.4f",
                    (unsigned long long)prog, index, (unsigned long long)n, changed, (double)g0.V[0], (double)g0.V[1], (double)g0.V[2], (double)g0.C[3], (double)g0.C[7],
                    (double)g0.C[11], (double)g0.P[0], d->has_prev ? "have" : "none", (double)g1.V[0], (double)g1.V[1], (double)g1.V[2], (double)g1.C[3], (double)g1.C[7],
                    (double)g1.C[11], (double)d->P[0][0]);
    }
    static const char* names[] = {"", "projection", "view-projection", "transposed view-projection", "rotation"};
    int shown = 0;
    for (size_t o = 0; o + 16 <= nf && shown < 6; o += 4)
        if (const int kd = diag_kind(f + o, d->game)) {
            shown++;
            if (may_tell()) mkwvk::logf("%s prog %016llx stage %d uniform buffer %u (%llu bytes): %s-like at +%zu: %.4f %.4f %.4f %.4f / %.4f %.4f %.4f %.4f / %.4f %.4f %.4f %.4f / %.4f %.4f %.4f %.4f",
                        tag, (unsigned long long)prog, stage, index, (unsigned long long)n, names[kd], o * 4, (double)f[o], (double)f[o + 1], (double)f[o + 2],
                        (double)f[o + 3], (double)f[o + 4], (double)f[o + 5], (double)f[o + 6], (double)f[o + 7], (double)f[o + 8], (double)f[o + 9],
                        (double)f[o + 10], (double)f[o + 11], (double)f[o + 12], (double)f[o + 13], (double)f[o + 14], (double)f[o + 15]);
        }
    if (!shown && !compute) mkwvk::logf("vr-diag-ubo: prog %016llx stage %d uniform buffer %u (%llu bytes): nothing camera-like", (unsigned long long)prog, stage, index, (unsigned long long)n);
    if (std::fabs(d->game.P[0]) > 1e-3f && std::fabs(d->game.P[5]) > 1e-3f) {   // the game's field of view as lone numbers
        const float src[4] = {d->game.P[0], d->game.P[5], 1.f / d->game.P[0], 1.f / d->game.P[5]};
        static const char* sn[4] = {"projection x scale", "projection y scale", "1 / x scale", "1 / y scale"};
        int hits = 0;
        for (size_t i = 0; i < nf && hits < 8; i++)
            for (int j = 0; j < 4; j++)
                if (std::fabs(f[i] - src[j]) <= 1e-5f * std::fabs(src[j])) {
                    hits++;
                    if (may_tell()) mkwvk::logf("%s prog %016llx stage %d uniform buffer %u: the game's %s (%.5f) at +%zu", tag, (unsigned long long)prog, stage, index, sn[j], (double)f[i], i * 4);
                    break;
                }
    }
}
int64_t diag_patched() { return diag_total.load(); }
void diag_log(uint64_t prog, const char* what, int64_t patched) {
    static std::mutex mu;
    static std::map<uint64_t, int> seen;
    std::lock_guard lk(mu);
    if (seen[prog]++ >= 3) return;
    mkwvk::logf("vr-diag: prog %016llx:%s; camera matrices changed in its uniform buffers: %lld", (unsigned long long)prog, what, (long long)patched);
}
bool stereo_left_done() { return st.left_done; }
void stereo_prepass() {   // (the first submit's memory is in: the camera is looked for where it was before. Found: the game's own replay is the left eye)
    if (!stereo_active() || st.have.load()) return;
    std::vector<std::pair<uint64_t, uint64_t>> hints;
    {
        std::lock_guard lk(st.mu);
        hints = st.hints;
    }
    static thread_local std::vector<uint8_t> buf;
    for (const auto& h : hints) {
        buf.resize(h.second);
        if (!ub_read(h.first, buf.data(), h.second)) continue;
        Cam c;
        if (!find_camera(reinterpret_cast<const float*>(buf.data()), h.second / 4, c)) continue;
        std::lock_guard lk(st.mu);
        st.eyes = make_eyes(c, st.last);
        st.have = true;
        st.eye = st.aer_eye;   // (alternate-eye rendering: the game's own replay of this frame is drawn for this eye)
        st.left_done = false;
        return;
    }
}

int patch_uniform(uint8_t* data, uint64_t n, uint64_t addr) {
    const int mode = stereo_mode();
    if (!mode) return 0;
    float* f = reinterpret_cast<float*>(data);
    const size_t nf = n / 4;
    if (mode == 1) {
        Cam c;
        if (!find_camera(f, nf, c)) return 0;
        std::lock_guard lk(st.mu);
        if (st.have.load()) return 0;
        st.eyes = make_eyes(c, st.last);
        st.have = true;
        {   // (where it was: the next frames look there first, before their draws)
            const std::pair<uint64_t, uint64_t> h{addr, n};
            auto& v = st.hints;
            v.erase(std::remove(v.begin(), v.end(), h), v.end());
            v.insert(v.begin(), h);
            if (v.size() > 24) v.resize(24);
        }
        if (st.frames < 3)
            mkwvk::logf("vr: stereo: the game's camera found: projection %g x %g, at (%.1f, %.1f, %.1f)%s", c.P[0], c.P[5], c.C[3],
                        c.C[7], c.C[11], st.last.valid ? "" : " (no frame before it yet)");
        return 0;
    }
    std::shared_ptr<const Eyes> d;
    {
        std::lock_guard lk(st.mu);
        d = st.eyes;
    }
    const int k = st.eye.load();
    if (!d || k < 0) return 0;
    const int r = patch_buffer(f, nf, *d, k);
    if (r > 0) st.patched = true, diag_total += r;
    if (r > 0) {   // (where the game keeps its camera: each uniform buffer address once)
        static std::mutex am;
        static std::set<uint64_t> told;
        std::lock_guard lk(am);
        if (told.size() < 40 && told.insert(addr).second)
            mkwvk::logf("vr-diag-cam: camera matrices in the uniform buffer at %#llx +%#llx (%d changed)", (unsigned long long)addr, (unsigned long long)n, r);
    }
    return r;
}

bool stereo_begin() {
    if (!stereo_active()) return false;
    poll_events();
    if (!g.running) return false;
    XrFrameState fs{XR_TYPE_FRAME_STATE};
    XrFrameWaitInfo wi{XR_TYPE_FRAME_WAIT_INFO};
    if (!XR_CHK(xrWaitFrame(g.session, &wi, &fs))) return false;
    {
        std::lock_guard qk(dev.present_mu);
        XrFrameBeginInfo bi{XR_TYPE_FRAME_BEGIN_INFO};
        if (!XR_CHK(xrBeginFrame(g.session, &bi))) return false;
    }
    if (!fs.shouldRender) {
        end_frame_empty(fs.predictedDisplayTime);
        return false;
    }
    st.time = fs.predictedDisplayTime;
    st.period = fs.predictedDisplayPeriod;
    st.copied[0] = st.copied[1] = false;
    {   // (the headset's eye distance: the eyes' separation in the game follows it)
        XrViewLocateInfo li{XR_TYPE_VIEW_LOCATE_INFO};
        li.viewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
        li.displayTime = st.time;
        li.space = g.space;
        XrViewState vs{XR_TYPE_VIEW_STATE};
        XrView v[2] = {{XR_TYPE_VIEW}, {XR_TYPE_VIEW}};
        uint32_t n = 0;
        if (xrLocateViews && XR_SUCCEEDED(xrLocateViews(g.session, &li, &vs, 2, &n, v)) && n == 2) {
            const float dx = v[1].pose.position.x - v[0].pose.position.x, dy = v[1].pose.position.y - v[0].pose.position.y,
                        dz = v[1].pose.position.z - v[0].pose.position.z;
            const float d = std::sqrt(dx * dx + dy * dy + dz * dz);
            if (d > 0.04f && d < 0.09f) {
                st.ipd = d;
                if (!st.sep_fixed) st.sep = st.ipd * st.conv / g.distance;
            }
        }
    }
    return true;
}

bool stereo_copy_eye(int e, VkImage src, uint32_t sw, uint32_t sh, const int crop[4], VkImageLayout src_layout) {
    if (e < 0 || e > 1 || !src || g.dead) return false;
    int x0 = 0, y0 = 0, x1 = (int)sw, y1 = (int)sh;
    if (crop && crop[2] > 0 && crop[3] > 0) x0 = crop[0], y0 = crop[1], x1 = std::min(crop[0] + crop[2], (int)sw), y1 = std::min(crop[1] + crop[3], (int)sh);
    const uint32_t cw = (uint32_t)std::max(1, x1 - x0), ch = (uint32_t)std::max(1, y1 - y0);
    if (!ensure_eyes(cw, ch)) return false;

    uint32_t index = 0;
    XrSwapchainImageAcquireInfo ai{XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO};
    {
        std::lock_guard qk(dev.present_mu);
        if (!XR_CHK(xrAcquireSwapchainImage(st.sc[e], &ai, &index))) return false;
    }
    XrSwapchainImageWaitInfo wsi{XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO};
    wsi.timeout = XR_INFINITE_DURATION;
    const bool waited = XR_CHK(xrWaitSwapchainImage(st.sc[e], &wsi));
    bool submitted = false;
    if (waited && index < st.images[e].size()) {
        Slot& s = g.slots[g.slot];
        g.slot = (g.slot + 1) % 3;
        vkWaitForFences(dev.device, 1, &s.fence, VK_TRUE, UINT64_MAX);
        vkResetCommandPool(dev.device, s.pool, 0);
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(s.cmd, &bi);
        VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        mb.srcStageMask = mb.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        mb.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
        VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dep.memoryBarrierCount = 1;
        dep.pMemoryBarriers = &mb;
        vkCmdPipelineBarrier2(s.cmd, &dep);
        VkImage dst = st.images[e][index];
        barrier(s.cmd, dst, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0,
                VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
        if (g.raw_copy) {
            VkImageCopy c{};
            c.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            c.srcOffset = {x0, y0, 0};
            c.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            c.extent = {cw, ch, 1};
            vkCmdCopyImage(s.cmd, src, src_layout, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &c);
        } else {
            VkImageBlit b{};
            b.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            b.srcOffsets[0] = {x0, y0, 0};
            b.srcOffsets[1] = {x1, y1, 1};
            b.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            b.dstOffsets[1] = {(int32_t)cw, (int32_t)ch, 1};
            vkCmdBlitImage(s.cmd, src, src_layout, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &b, VK_FILTER_LINEAR);
        }
        barrier(s.cmd, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
        vkEndCommandBuffer(s.cmd);
        VkCommandBufferSubmitInfo cb{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
        cb.commandBuffer = s.cmd;
        VkSubmitInfo2 si{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
        si.commandBufferInfoCount = 1;
        si.pCommandBufferInfos = &cb;
        std::lock_guard qk(dev.present_mu);
        queue_flush_locked();
        vkResetFences(dev.device, 1, &s.fence);
        if (vkQueueSubmit2(dev.queue, 1, &si, s.fence) == VK_SUCCESS) {
            submitted = true;
        } else {
            mkwvk::logf("vr: the stereo copy's submit failed: VR is off");
            g.dead = true;
            vkQueueSubmit2(dev.queue, 0, nullptr, s.fence);
        }
    }
    std::lock_guard qk(dev.present_mu);
    XrSwapchainImageReleaseInfo ri{XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO};
    xrReleaseSwapchainImage(st.sc[e], &ri);
    st.copied[e] = submitted;
    return submitted;
}

// alternate-eye rendering: the game drew this frame once, for one eye (its camera changed in patch_uniform). Its picture goes
// into that eye's cache; both eyes' latest pictures go to the headset (the other eye's is the frame before's).
bool stereo_cache_write(int e, VkImage src, int x0, int y0, uint32_t cw, uint32_t ch) {
    if (e < 0 || e > 1 || !src || !st.cimg[e] || g.dead) return false;
    Slot& s = g.slots[g.slot];
    g.slot = (g.slot + 1) % 3;
    vkWaitForFences(dev.device, 1, &s.fence, VK_TRUE, UINT64_MAX);
    vkResetCommandPool(dev.device, s.pool, 0);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(s.cmd, &bi);
    VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    mb.srcStageMask = mb.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    mb.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
    VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dep.memoryBarrierCount = 1;
    dep.pMemoryBarriers = &mb;
    vkCmdPipelineBarrier2(s.cmd, &dep);
    VkImage dst = st.cimg[e];
    barrier(s.cmd, dst, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0,
            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    if (g.raw_copy) {
        VkImageCopy c{};
        c.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        c.srcOffset = {x0, y0, 0};
        c.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        c.extent = {cw, ch, 1};
        vkCmdCopyImage(s.cmd, src, VK_IMAGE_LAYOUT_GENERAL, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &c);
    } else {
        VkImageBlit b{};
        b.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        b.srcOffsets[0] = {x0, y0, 0};
        b.srcOffsets[1] = {x0 + (int32_t)cw, y0 + (int32_t)ch, 1};
        b.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        b.dstOffsets[1] = {(int32_t)cw, (int32_t)ch, 1};
        vkCmdBlitImage(s.cmd, src, VK_IMAGE_LAYOUT_GENERAL, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &b, VK_FILTER_NEAREST);
    }
    barrier(s.cmd, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            VK_ACCESS_2_TRANSFER_READ_BIT);
    vkEndCommandBuffer(s.cmd);
    VkCommandBufferSubmitInfo cb{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    cb.commandBuffer = s.cmd;
    VkSubmitInfo2 si{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    si.commandBufferInfoCount = 1;
    si.pCommandBufferInfos = &cb;
    std::lock_guard qk(dev.present_mu);
    queue_flush_locked();
    vkResetFences(dev.device, 1, &s.fence);
    if (vkQueueSubmit2(dev.queue, 1, &si, s.fence) != VK_SUCCESS) {
        mkwvk::logf("vr: the stereo cache copy's submit failed: VR is off");
        g.dead = true;
        vkQueueSubmit2(dev.queue, 0, nullptr, s.fence);
        return false;
    }
    return true;
}


// ---- hookshot_vr_dump.txt: both cached eye pictures, side by side, in hookshot_vr_eyes.png (half size) ---------------------
namespace {
uint32_t crc_of(const uint8_t* p, size_t n, uint32_t crc = 0) {
    static uint32_t t[256];
    static bool init = false;
    if (!init) {
        for (uint32_t i = 0; i < 256; i++) {
            uint32_t v = i;
            for (int k = 0; k < 8; k++) v = (v & 1) ? 0xEDB88320u ^ (v >> 1) : v >> 1;
            t[i] = v;
        }
        init = true;
    }
    crc = ~crc;
    for (size_t i = 0; i < n; i++) crc = t[(crc ^ p[i]) & 255] ^ (crc >> 8);
    return ~crc;
}
void put32(std::vector<uint8_t>& v, uint32_t x) { for (int i = 3; i >= 0; i--) v.push_back((uint8_t)(x >> (i * 8))); }
void png_chunk(std::vector<uint8_t>& out, const char* type, const std::vector<uint8_t>& data) {
    put32(out, (uint32_t)data.size());
    std::vector<uint8_t> td(type, type + 4);
    td.insert(td.end(), data.begin(), data.end());
    out.insert(out.end(), td.begin(), td.end());
    put32(out, crc_of(td.data(), td.size()));
}
bool write_png(const char* path, uint32_t w, uint32_t h, const std::vector<uint8_t>& rgb) {
    std::vector<uint8_t> raw;
    raw.reserve((size_t)h * (w * 3 + 1));
    for (uint32_t y = 0; y < h; y++) {
        raw.push_back(0);
        raw.insert(raw.end(), rgb.begin() + (size_t)y * w * 3, rgb.begin() + (size_t)(y + 1) * w * 3);
    }
    std::vector<uint8_t> z{0x78, 0x01};
    uint32_t a = 1, b = 0;
    for (uint8_t v : raw) { a = (a + v) % 65521; b = (b + a) % 65521; }
    for (size_t o = 0; o < raw.size(); o += 65535) {
        const size_t n = std::min<size_t>(65535, raw.size() - o);
        z.push_back(o + n >= raw.size() ? 1 : 0);
        z.push_back((uint8_t)(n & 255)), z.push_back((uint8_t)(n >> 8));
        z.push_back((uint8_t)(~n & 255)), z.push_back((uint8_t)((~n >> 8) & 255));
        z.insert(z.end(), raw.begin() + o, raw.begin() + o + n);
    }
    put32(z, (b << 16) | a);
    std::vector<uint8_t> out{0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A}, ih;
    put32(ih, w), put32(ih, h);
    ih.push_back(8), ih.push_back(2), ih.push_back(0), ih.push_back(0), ih.push_back(0);
    png_chunk(out, "IHDR", ih);
    png_chunk(out, "IDAT", z);
    png_chunk(out, "IEND", {});
    FILE* f = fopen(path, "wb");
    if (!f) return false;
    fwrite(out.data(), 1, out.size(), f);
    fclose(f);
    return true;
}
void dump_eyes(uint32_t w, uint32_t h) {
    const VkDeviceSize one = (VkDeviceSize)w * h * 4, total = one * 2;
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkBufferCreateInfo bc{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bc.size = total;
    bc.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bc.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(dev.device, &bc, nullptr, &buf) != VK_SUCCESS) return;
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(dev.device, buf, &mr);
    uint32_t type = UINT32_MAX;
    for (uint32_t i = 0; i < dev.memory.memoryTypeCount; i++) {
        const VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        if ((mr.memoryTypeBits & (1u << i)) && (dev.memory.memoryTypes[i].propertyFlags & want) == want) { type = i; break; }
    }
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = mr.size;
    ai.memoryTypeIndex = type;
    if (type == UINT32_MAX || vkAllocateMemory(dev.device, &ai, nullptr, &mem) != VK_SUCCESS || vkBindBufferMemory(dev.device, buf, mem, 0) != VK_SUCCESS) {
        vkDestroyBuffer(dev.device, buf, nullptr);
        if (mem) vkFreeMemory(dev.device, mem, nullptr);
        mkwvk::logf("vr: eye dump: no memory for the copy");
        return;
    }
    Slot& s = g.slots[g.slot];
    g.slot = (g.slot + 1) % 3;
    vkWaitForFences(dev.device, 1, &s.fence, VK_TRUE, UINT64_MAX);
    vkResetCommandPool(dev.device, s.pool, 0);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(s.cmd, &bi);
    for (int e = 0; e < 2; e++) {
        VkBufferImageCopy r{};
        r.bufferOffset = one * e;
        r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        r.imageExtent = {w, h, 1};
        vkCmdCopyImageToBuffer(s.cmd, st.cimg[e], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buf, 1, &r);
    }
    vkEndCommandBuffer(s.cmd);
    VkCommandBufferSubmitInfo cb{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    cb.commandBuffer = s.cmd;
    VkSubmitInfo2 si{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    si.commandBufferInfoCount = 1;
    si.pCommandBufferInfos = &cb;
    {
        std::lock_guard qk(dev.present_mu);
        queue_flush_locked();
        vkResetFences(dev.device, 1, &s.fence);
        if (vkQueueSubmit2(dev.queue, 1, &si, s.fence) != VK_SUCCESS) {
            mkwvk::logf("vr: eye dump: submit failed");
            vkDestroyBuffer(dev.device, buf, nullptr);
            vkFreeMemory(dev.device, mem, nullptr);
            return;
        }
    }
    vkWaitForFences(dev.device, 1, &s.fence, VK_TRUE, UINT64_MAX);
    void* map = nullptr;
    if (vkMapMemory(dev.device, mem, 0, total, 0, &map) == VK_SUCCESS) {
        const uint8_t* px = static_cast<const uint8_t*>(map);
        const bool bgr = g.format == 44 || g.format == 50;
        const uint32_t hw = w / 2, hh = h / 2;
        std::vector<uint8_t> rgb((size_t)hw * 2 * hh * 3);
        for (int e = 0; e < 2; e++)
            for (uint32_t y = 0; y < hh; y++)
                for (uint32_t x = 0; x < hw; x++) {
                    const uint8_t* p = px + one * e + ((size_t)(y * 2) * w + x * 2) * 4;
                    uint8_t* o = &rgb[((size_t)y * hw * 2 + (size_t)e * hw + x) * 3];
                    o[0] = bgr ? p[2] : p[0], o[1] = p[1], o[2] = bgr ? p[0] : p[2];
                }
        const bool ok = write_png("hookshot_vr_eyes.png", hw * 2, hh, rgb);
        mkwvk::logf("vr: eye dump: %s hookshot_vr_eyes.png (left eye | right eye, %ux%u each, half size, format %d)", ok ? "wrote" : "could not write", hw, hh, (int)g.format);
    }
    vkDestroyBuffer(dev.device, buf, nullptr);
    vkFreeMemory(dev.device, mem, nullptr);
}
}

bool stereo_present(VkImage src, uint32_t sw, uint32_t sh, const int crop[4]) {
    if (!stereo_active() || !src || g.dead) return false;
    const bool fresh = stereo_ready();
    if (fresh) {
        st.stale = 0;
        ++st.nfresh;
    } else {   // (this frame was not drawn with an eye's camera: the latest pictures go again, a few times at most)
        if (!st.cvalid[0] || !st.cvalid[1]) return false;
        if (++st.stale > 8) {
            st.cvalid[0] = st.cvalid[1] = false;
            return false;
        }
        ++st.nstale;
    }
    if (((st.nfresh + st.nstale) % 240) == 1)
        mkwvk::logf("vr: stereo frames so far: %lld drawn for an eye, %lld sent again from the cache", (long long)st.nfresh, (long long)st.nstale);
    int x0 = 0, y0 = 0, x1 = (int)sw, y1 = (int)sh;
    if (crop && crop[2] > 0 && crop[3] > 0) x0 = crop[0], y0 = crop[1], x1 = std::min(crop[0] + crop[2], (int)sw), y1 = std::min(crop[1] + crop[3], (int)sh);
    const uint32_t cw = (uint32_t)std::max(1, x1 - x0), ch = (uint32_t)std::max(1, y1 - y0);
    if (!ensure_eyes(cw, ch) || !ensure_cache(cw, ch)) return false;
    if (!stereo_begin()) return true;   // (the headset's frame is not wanted now: handled)
    const int eye = st.aer_eye;
    if (fresh && stereo_cache_write(eye, src, x0, y0, cw, ch)) {
        st.cvalid[eye] = true;
        std::shared_ptr<const Eyes> dd;
        {
            std::lock_guard lk(st.mu);
            dd = st.eyes;
        }
        if (dd) {
            st.cfx[eye] = dd->P[eye][0];
            st.cfy[eye] = dd->P[eye][5];
            st.csep[eye] = dd->sep;
            st.cpose_ok[eye] = dd->has_pose;
            if (dd->has_pose) {
                const float* r = dd->raw[eye];
                st.cpose[eye].position = {r[0], r[1], r[2]};
                st.cpose[eye].orientation = {r[3], r[4], r[5], r[6]};
            }
            if (std::fabs(dd->P[eye][0] - st.last_logged_sx) > 0.01f) {
                mkwvk::logf("vr: eye picture drawn with projection sx %.4f sy %.4f (picture %ux%u; the game's own %.4f x %.4f)", (double)dd->P[eye][0], (double)dd->P[eye][5], cw, ch, (double)dd->game.P[0], (double)dd->game.P[5]);
                st.last_logged_sx = dd->P[eye][0];
            }
        }
    }
    if (!st.cvalid[0] || !st.cvalid[1]) {   // (the other eye's first picture comes with the next frame)
        end_frame_empty(st.time);
        st.handled = true;
        return true;
    }
    if (st.dump_req.exchange(false)) dump_eyes(cw, ch);
    for (int e = 0; e < 2; e++) stereo_copy_eye(e, st.cimg[e], cw, ch, nullptr, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    stereo_sync();
    stereo_end();
    return true;
}

void stereo_end() {
    const bool both = st.copied[0] && st.copied[1];
    std::shared_ptr<const Eyes> d;
    {
        std::lock_guard lk(st.mu);
        d = st.eyes;
    }
    // Each eye's picture as a projection layer. The pictures were drawn from eyes sep/2 either side of the middle, looking at a
    // window that is the game's own field of view, conv away (in game units), so the layer says the same: each eye is that far
    // out from the middle (in metres: the window is g.distance away), and sees the window's edges at those angles.
    XrCompositionLayerProjectionView pv[2];
    for (int e = 0; e < 2; e++) {
        pv[e] = XrCompositionLayerProjectionView{XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW};
        const float sx = st.cfx[e], sy = st.cfy[e];
        const float sepu = (e == 0 ? -0.5f : 0.5f) * (st.csep[e] != 0.f ? st.csep[e] : (d ? d->sep : st.sep));
        const float tx = 1.f / sx, ty = 1.f / sy;
        const bool tracked = st.cpose_ok[e];
        if (tracked) {   // (drawn from the headset's own eye, so shown from there: the headset moves it on to where the head is by then)
            pv[e].pose = st.cpose[e];
        } else {
            pv[e].pose.orientation.w = 1.0f;
            pv[e].pose.position = {sepu * g.distance / st.conv, 0.0f, 0.0f};
        }
        static const bool offaxis = [] { const char* e2 = getenv("HOOKSHOT_VR_OFFAXIS"); return e2 && *e2 && *e2 != '0'; }();
        const float shift = (offaxis && !tracked) ? sepu / st.conv : 0.f;
        pv[e].fov.angleLeft = std::atan(-tx - shift);
        pv[e].fov.angleRight = std::atan(tx - shift);
        pv[e].fov.angleUp = std::atan(ty);
        pv[e].fov.angleDown = -std::atan(ty);
        pv[e].subImage.swapchain = st.sc[e];
        pv[e].subImage.imageRect.offset = {0, 0};
        pv[e].subImage.imageRect.extent = {(int32_t)st.w, (int32_t)st.h};
        pv[e].subImage.imageArrayIndex = 0;
    }
    XrCompositionLayerProjection layer{XR_TYPE_COMPOSITION_LAYER_PROJECTION};
    layer.layerFlags = 0;
    layer.space = g.space;
    layer.viewCount = 2;
    layer.views = pv;
    const XrCompositionLayerBaseHeader* layers[1] = {(const XrCompositionLayerBaseHeader*)&layer};
    std::lock_guard qk(dev.present_mu);
    XrFrameEndInfo ei{XR_TYPE_FRAME_END_INFO};
    ei.displayTime = st.time;
    ei.environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;
    ei.layerCount = both && d ? 1 : 0;
    ei.layers = both && d ? layers : nullptr;
    XR_CHK(xrEndFrame(g.session, &ei));
    st.handled = true;
    const auto now = std::chrono::steady_clock::now();
    if (st.last_end.time_since_epoch().count())
        st.ms_sum += std::chrono::duration<double, std::milli>(now - st.last_end).count();
    st.last_end = now;
    ++st.frames;
    if (st.frames <= 2 || st.frames % 120 == 0) {
        mkwvk::logf("vr: stereo frame %lld sent: %.1f ms apart on average (the headset wants %.1f); eyes %.3f m apart; matrices changed so "
                    "far: view %lld, inverse %lld, projection %lld, product %lld, column-major %lld, rotation-only %lld/%lld, position %lld, other-depth projection %lld, product %lld, camera-relative %lld, direction vectors %lld, nearby copies %lld, field-of-view numbers %lld%s",
                    (long long)st.frames, st.frames > 1 ? st.ms_sum / (double)(st.frames - 1) : 0.0, st.period / 1e6, st.ipd,
                    (long long)st.counts[0], (long long)st.counts[1], (long long)st.counts[2], (long long)st.counts[3],
                    (long long)st.counts[4], (long long)st.counts[5], (long long)st.counts[6], (long long)st.counts[7], (long long)st.counts[8], (long long)st.counts[9], (long long)st.counts[10], (long long)st.counts[11], (long long)st.counts[12], (long long)st.counts[13], both ? "" : " (a copy failed)");
        if (st.frames % 120 == 0) st.ms_sum = 0, st.frames = 1;
    }
}

void shutdown() {
    if (g.created && dev.device) {
        vkDeviceWaitIdle(dev.device);
        destroy_swapchain();
        destroy_eyes();
        destroy_cache();
        for (Slot& s : g.slots) {
            if (s.fence) vkDestroyFence(dev.device, s.fence, nullptr);
            if (s.pool) vkDestroyCommandPool(dev.device, s.pool, nullptr);
        }
    }
    if (g.space) xrDestroySpace(g.space);
    if (g.session) xrDestroySession(g.session);
    if (g.instance) xrDestroyInstance(g.instance);
    g = State{};
}


// ---- camera probe (HOOKSHOT_VR_PROBE=1) --------------------------------------------------------------------------------
// Finds where the game keeps its camera. At a few frames, every uniform buffer bound at a draw is scanned for 4x4 float
// blocks that look like a perspective projection or a rigid (rotation + position) view matrix. What turns up in most draws
// of a frame is the camera. The result goes to the log as "vr-probe:" lines.
namespace {
struct Cand {
    float m[16];
    uint64_t addr;
    uint32_t off;
    int kind;   // 1 projection, 2 rigid
    uint32_t count;
    int stage, slot;
    uint64_t prog;
};
std::mutex probe_mu;
std::atomic<uint64_t> probe_frames{0};
uint64_t probe_cur = ~0ull;
int probe_done = 0;
std::map<std::string, Cand> probe_map;
uint64_t probe_draws = 0, probe_binds = 0, probe_blocks = 0, probe_beat = 0, probe_fdraws = 0, probe_prev_fdraws = 0;
bool probe_on() {
    static const bool on = [] { const char* e = getenv("HOOKSHOT_VR_PROBE"); return e && *e && *e != '0'; }();
    return on;
}

bool probe_wanted(uint64_t f) {
    static const bool on = [] { const char* e = getenv("HOOKSHOT_VR_PROBE"); return e && *e && *e != '0'; }();
    // (a frame that follows a busy one, so not the title, the loading or the menus; one frame in 20, 30 of them in all)
    return on && probe_done < 30 && probe_prev_fdraws > 800 && f % 20 == 0;
}
bool fin(const float* m) {
    for (int i = 0; i < 16; i++)
        if (!(std::fabs(m[i]) < 1e7f)) return false;
    return true;
}
bool nearf(float a, float b, float t = 1e-3f) { return std::fabs(a - b) <= t; }
bool rigid(const float* m, bool rows) {   // (the 3x3 inside is a rotation, whether the vectors are rows or columns)
    float v[3][3];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) v[i][j] = rows ? m[i * 4 + j] : m[j * 4 + i];
    for (int i = 0; i < 3; i++) {
        const float l = v[i][0] * v[i][0] + v[i][1] * v[i][1] + v[i][2] * v[i][2];
        if (!nearf(l, 1.f, 5e-3f)) return false;
        for (int k = i + 1; k < 3; k++)
            if (std::fabs(v[i][0] * v[k][0] + v[i][1] * v[k][1] + v[i][2] * v[k][2]) > 5e-3f) return false;
    }
    return true;
}
bool unit3(float a, float b, float c) { return nearf(a * a + b * b + c * c, 1.f, 1e-2f); }
bool viewproj(const float* m) {   // (view x projection: its w row is the camera's forward vector, unit length; rows or columns)
    int nz = 0;
    for (int i = 0; i < 16; i++) nz += m[i] != 0.f;
    if (nz < 10) return false;
    const bool col = unit3(m[3], m[7], m[11]) && std::fabs(m[15]) < 1e5f && (m[3] != 0 || m[7] != 0 || m[11] != 0);
    const bool row = unit3(m[12], m[13], m[14]) && std::fabs(m[15]) < 1e5f && (m[12] != 0 || m[13] != 0 || m[14] != 0);
    return col || row;
}
bool projection(const float* m) {   // (a perspective: -1 or 1 in the w row/column's z place, 0 in the corner, x and y scales set)
    if (m[0] == 0 || m[5] == 0) {
        if (m[0] == 0 && m[1] == 0) return false;
    }
    const bool a = nearf(std::fabs(m[11]), 1.f) && nearf(m[15], 0.f) && m[3] == 0 && m[7] == 0;     // (column-major, w from z)
    const bool b = nearf(std::fabs(m[14]), 1.f) && nearf(m[15], 0.f) && m[12] == 0 && m[13] == 0;   // (row-major)
    return a || b;
}
void probe_flush_locked() {
    if (probe_map.empty()) return;
    std::vector<const Cand*> v;
    for (auto& kv : probe_map) v.push_back(&kv.second);
    std::sort(v.begin(), v.end(), [](const Cand* a, const Cand* b) { return a->count > b->count; });
    mkwvk::logf("vr-probe: ---- frame %llu: %zu distinct candidates", (unsigned long long)probe_cur, v.size());
    int np = 0, nr = 0, nv = 0;
    for (const Cand* c : v) {
        int& n = c->kind == 1 ? np : c->kind == 2 ? nr : nv;
        if (n++ >= 14) continue;
        mkwvk::logf("vr-probe: %s x%u prog %016llx s%d ubo%d %#llx+%#x: %g %g %g %g | %g %g %g %g | %g %g %g %g | %g %g %g %g",
                    c->kind == 1 ? "PROJ " : c->kind == 2 ? "RIGID" : "VP   ", c->count, (unsigned long long)c->prog, c->stage, c->slot,
                    (unsigned long long)c->addr, c->off, c->m[0], c->m[1], c->m[2], c->m[3], c->m[4], c->m[5], c->m[6], c->m[7],
                    c->m[8], c->m[9], c->m[10], c->m[11], c->m[12], c->m[13], c->m[14], c->m[15]);
    }
    probe_map.clear();
    probe_done++;
}
} // namespace

void probe_tick() { probe_frames.fetch_add(1, std::memory_order_relaxed); }

void probe_draw(uint64_t prog, const NvnState& s) {
    if (!probe_on()) return;
    const uint64_t f = probe_frames.load(std::memory_order_relaxed);
    {
        std::lock_guard lk(probe_mu);
        if (f != probe_cur) {
            probe_flush_locked();
            probe_cur = f;
            probe_prev_fdraws = probe_fdraws;
            probe_fdraws = 0;
        }
        probe_fdraws++;
        probe_draws++;
        if (f / 300 != probe_beat) {
            probe_beat = f / 300;
            mkwvk::logf("vr-probe: alive at frame %llu: %llu draws, %llu buffers read, %llu 4x4 blocks looked at since the last note "
                        "(%llu draws in the last frame; %d of 30 frames scanned)", (unsigned long long)f, (unsigned long long)probe_draws,
                        (unsigned long long)probe_binds, (unsigned long long)probe_blocks, (unsigned long long)probe_prev_fdraws,
                        probe_done);
            probe_draws = probe_binds = probe_blocks = 0;
        }
    }
    if (!probe_wanted(f)) return;
    static thread_local std::vector<float> buf;
    for (int st = 0; st < kNvnStages; st++)
        for (int sl = 0; sl < 32; sl++) {
            const bool ssbo = sl >= 16;
            const auto& b = ssbo ? s.ssbo[st][sl - 16] : s.ubo[st][sl];
            if (!b.addr || b.size < 64) continue;
            const uint64_t n = std::min<uint64_t>(b.size, 0x2000) & ~15ull;
            buf.assign(n / 4, 0.f);
            if (!(ssbo ? memory_read_as_submitted(b.addr, buf.data(), n) : ub_read(b.addr, buf.data(), n))) continue;
            std::lock_guard lk(probe_mu);
            probe_binds++;
            for (uint64_t o = 0; o + 64 <= n; o += 16) {
                const float* m = &buf[o / 4];
                probe_blocks++;
                if (!fin(m)) continue;
                bool plain = true;   // (all 0, 1 or -1: constants and padding, not a camera)
                for (int i = 0; i < 16; i++)
                    if (m[i] != 0.f && m[i] != 1.f && m[i] != -1.f) { plain = false; break; }
                if (plain) continue;
                int kind = 0;
                if (projection(m)) kind = 1;
                else if ((rigid(m, true) || rigid(m, false)) && (m[0] != 0 || m[1] != 0 || m[2] != 0)) kind = 2;
                else if (viewproj(m)) kind = 3;
                if (!kind) continue;
                char key[200];
                snprintf(key, sizeof key, "%d:%g:%g:%g:%g:%g:%g", kind, m[0], m[1], m[2], m[5], m[10], m[12]);
                auto it = probe_map.find(key);
                if (it != probe_map.end()) { it->second.count++; continue; }
                if (probe_map.size() > 6000) continue;
                Cand c{};
                std::memcpy(c.m, m, 64);
                c.addr = b.addr, c.off = (uint32_t)o, c.kind = kind, c.count = 1, c.stage = st, c.slot = sl, c.prog = prog;
                probe_map.emplace(key, c);
            }
        }
}

} // namespace vr
} // namespace mkwvk

#endif // HOOKSHOT_VR_BUILD
