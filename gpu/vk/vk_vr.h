// VR for Hookshot (cinema mode): the game's finished frame shown on a floating screen in an OpenXR headset
// (Quest 3 through SteamVR, or any OpenXR runtime with XR_KHR_vulkan_enable2). Implemented in vk_vr.cpp.
//
// Switched on by HOOKSHOT_VR=1 (put it in hookshot_env.txt). Built in only when the build was given OPENXR_ROOT
// (nx2/build.py defines HOOKSHOT_VR_BUILD); without it every call here is an empty stand-in and the renderer is unchanged.
// openxr_loader.dll is loaded at run time, like vulkan-1.dll: put it beside hookshot.exe.
#pragma once

#include "vk_loader.h"

namespace mkwvk {
struct NvnState;
namespace vr {

#ifdef HOOKSHOT_VR_BUILD
bool wanted();                       // HOOKSHOT_VR is set to something other than 0
bool init();                         // loads openxr_loader.dll, makes the XrInstance, finds the headset. false: run without VR
bool active();                       // init() succeeded
VkResult create_instance(const VkInstanceCreateInfo* info, VkInstance* out);              // instead of vkCreateInstance
VkPhysicalDevice physical_device(VkInstance instance);                                    // the GPU the headset is on
VkResult create_device(VkPhysicalDevice phys, const VkDeviceCreateInfo* info, VkDevice* out);   // instead of vkCreateDevice
bool create_session();               // once dev.device and dev.queue exist
// Each game frame, from present_image: waits for the headset's frame, copies src (the game's picture, VK_IMAGE_LAYOUT_GENERAL,
// cropped by crop like the window's present) onto the screen in the headset, and ends the frame. Never throws the game off:
// any failure turns VR off and the window carries on.
void present(VkImage src, uint32_t sw, uint32_t sh, const int crop[4]);
void shutdown();
// Stereo screen (HOOKSHOT_VR_STEREO=1): the game's frame is replayed once more for each eye with the camera's matrices changed
// in the uniform data, and each eye's picture goes to the headset's matching eye (vk_replay.cpp do_present, vk_draw.cpp).
bool stereo_active();                // on, and the headset is running
int stereo_mode();                   // 0 nothing to do, 1 looking for the game's camera, 2 changing it for stereo_set_eye's eye
bool stereo_ready();                 // this frame's camera was found
void stereo_frame_start();           // the game's frame begins (its first submit)
void stereo_frame_done();            // the game's frame is over: its camera becomes the "last one"
void stereo_set_eye(int eye);        // -1: none (the game's own camera), 0 left, 1 right
int patch_uniform(uint8_t* data, uint64_t n, uint64_t addr);
void stereo_prepass();               // the frame's first submit is in: look for the camera where it was before
bool stereo_left_done();             // the camera was found by then: the game's own replay was the left eye   // mode 1: look for the camera in these bytes; mode 2: change it. How many matrices changed
bool stereo_begin();                 // the headset's frame begins (waits for it). false: draw the frame as before
bool stereo_copy_eye(int eye, VkImage src, uint32_t sw, uint32_t sh, const int crop[4], VkImageLayout src_layout = VK_IMAGE_LAYOUT_GENERAL);
bool stereo_present(VkImage src, uint32_t sw, uint32_t sh, const int crop[4]);   // alternate-eye rendering: this frame's picture is one eye's; both eyes' latest go to the headset
void stereo_end();                   // both eyes in: the frame goes to the headset
void stereo_cost(double ms);          // how long the eyes took: too slow pauses stereo
bool skip_compute(uint64_t prog, uint32_t x, uint32_t y, uint32_t z);                 // the second eye's replay does not run the game's compute passes again (their results are the first eye's)
bool hide_draw(uint64_t prog, bool indirect, int64_t count, int instances);          // (a test) true: leave this draw out (hookshot_vr_hide.txt)
bool diag_scan_wanted(uint64_t prog);                                                // (a test) one of the foliage programs, while stereo draws
int diag_scan(const uint8_t* data, uint64_t n);                                      // (a test) how many camera-like windows these bytes hold
void diag_ubo(uint64_t prog, int stage, uint32_t index, const uint8_t* data, uint64_t n, bool compute = false, int changed = -1);   // (a test) what camera-like data a uniform buffer holds, once per slot
bool diag_on();                                                                      // (a test) stereo is running: indirect draws' programs are described in the log
int64_t diag_patched();                                                              // camera matrices changed so far
void diag_log(uint64_t prog, const char* what, int64_t patched);
void stereo_sync();                  // wait until the GPU has finished everything submitted so far (safety)
void probe_tick();                   // each presented frame (HOOKSHOT_VR_PROBE=1: where the game keeps its camera)
void probe_draw(uint64_t prog, const NvnState& s);   // each draw
#else
inline bool wanted() { return false; }
inline bool init() { return false; }
inline bool active() { return false; }
inline VkResult create_instance(const VkInstanceCreateInfo*, VkInstance*) { return VK_ERROR_INITIALIZATION_FAILED; }
inline VkPhysicalDevice physical_device(VkInstance) { return VK_NULL_HANDLE; }
inline VkResult create_device(VkPhysicalDevice, const VkDeviceCreateInfo*, VkDevice*) { return VK_ERROR_INITIALIZATION_FAILED; }
inline bool create_session() { return false; }
inline void present(VkImage, uint32_t, uint32_t, const int*) {}
inline void shutdown() {}
inline bool stereo_active() { return false; }
inline int stereo_mode() { return 0; }
inline bool stereo_ready() { return false; }
inline void stereo_frame_start() {}
inline void stereo_frame_done() {}
inline void stereo_set_eye(int) {}
inline int patch_uniform(uint8_t*, uint64_t, uint64_t) { return 0; }
inline void stereo_prepass() {}
inline bool stereo_left_done() { return false; }
inline bool stereo_begin() { return false; }
inline bool stereo_copy_eye(int, VkImage, uint32_t, uint32_t, const int*, VkImageLayout = VK_IMAGE_LAYOUT_GENERAL) { return false; }
inline bool stereo_present(VkImage, uint32_t, uint32_t, const int*) { return false; }
inline void stereo_end() {}
inline void stereo_sync() {}
inline bool hide_draw(uint64_t, bool, int64_t, int) { return false; }
inline bool diag_scan_wanted(uint64_t) { return false; }
inline int diag_scan(const uint8_t*, uint64_t) { return 0; }
inline void diag_ubo(uint64_t, int, uint32_t, const uint8_t*, uint64_t, bool = false, int = -1) {}
inline bool diag_on() { return false; }
inline int64_t diag_patched() { return 0; }
inline void diag_log(uint64_t, const char*, int64_t) {}
inline bool skip_compute(uint64_t, uint32_t, uint32_t, uint32_t) { return false; }
inline void stereo_cost(double) {}
inline void probe_tick() {}
inline void probe_draw(uint64_t, const NvnState&) {}
#endif

} // namespace vr
} // namespace mkwvk
