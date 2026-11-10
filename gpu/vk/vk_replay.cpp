// The NVN replay on Vulkan: the game's recorded command buffers (host/nvn_native.c, records in rx_nvn.h) at each
// submit, recorded into a Vulkan command buffer and submitted with the next value of a timeline semaphore (the NVN
// fences); and the window presents.
//
// Each recording has two command buffers: the replay's, and one submitted right before it that texture uploads and
// new images' layout changes go to (vk_texture.cpp), so no pass has to end for them. The stream's memory writes
// happen on the GPU timeline: guest memory is shared with the GPU, so a CPU write at replay time would reach draws
// recorded before it. Uniform updates (UPDATE_UB) are written at the end of their recording, and the draws after
// one bind a copy of the range with it in (ub_overlay), so a pass need not end for them either.
//
// Without a window (HOS_HEADLESS) presents render into an image of their own; RX_VK_SHOTS=<dir> writes every
// RX_VK_SHOT_EVERY-th one (default 300) there as a PNG.
#include <algorithm>
#include <sys/stat.h>
#include <atomic>
#include <chrono>
#include <cerrno>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <exception>
#include <thread>
#include <string>
#include <map>
#include <unordered_map>
#include <unordered_set>

#include "vk_temporal.h"
#include "vk_fsr3_inputs.h"   // (the camera jitter while AMD FSR 3.1 upscales)
#include "status_line.h"
#include "vk_reflex.h"
#include "vk_antilag.h"
#include "vk_draw.h"
#include "vk_dump.h"
#include "vk_paths.h"   // (game_module_path: the frame trace beside the game)
#include "vk_gprof.h"
#include "vk_vr.h"
#include "vk_framegen.h"
#include "vk_menu.h"
#include "vk_jobs.h"
#include "vk_shader.h"
#include "vk_state.h"
#include "vk_texture.h"
#include "vk_xfer.h"
#include "vk_present_rect.h"
#include "vk_present_trace.h"

#include "vk_internal.h"
#include "vk_work.h"   // (after the others: counts the commands that do GPU work, for full_barrier)

namespace mkwvk {
extern std::atomic<bool> movie_networks_slow;   // (vk_draw.cpp: the movie networks left out, issue 109)
int64_t presents_now;   // (the presents so far, for SlowWatch)
void hang_watch_start();   // (EvilOOT, vk_hangdump.cpp: RX_VK_HANG_DUMP)
struct CondNote { int64_t present; uint64_t addr, x, y; int mode; bool on; };   // (the last render conditions, for a dark frame)
CondNote cond_ring[1 << 15];
uint32_t cond_ring_at = 0;
namespace {

// ---- submissions: a timeline semaphore; value n = the n-th submit done ----
VkSemaphore timeline = VK_NULL_HANDLE;
// ---- (hookshot) recordings by number: a second timeline, signalled at the end of every recording with its number, and the
// recording each submission was finished in. The timeline above is signalled only as far as no older submission is stopped
// at a wait, and every NVN queue's submissions share its numbers: a fence taken on one queue waited for the other queues'
// stopped work too, and a wait for it forced that work past the waits it stopped at (OoT's presentation work run ahead of
// the scene work it waits for: a frame of the fog, haze and sun shafts of the frame before). NVN's fences are per queue: a
// wait for one is a wait for that queue's work, here for the recordings it was finished in. RX_VK_PER_QUEUE_FENCES=0: as
// before ----
VkSemaphore rec_timeline = VK_NULL_HANDLE;
uint64_t rec_seq = 0;                       // (the recordings submitted so far; replay_mu)
std::vector<uint64_t> rec_open_values;      // (the submissions finished in the recording being made; replay_mu)
std::mutex rec_map_mu;
std::map<uint64_t, uint64_t> value_rec;     // submission value (or a recording's own) -> the recording that finished it
std::unordered_map<uint64_t, uint64_t> q_last;   // queue -> its last submission's value (sync_order_mu)
std::map<uint64_t, uint64_t> value_queue;   // a submission's value -> its queue (sync_order_mu)
std::atomic<uint64_t> pushed{0};
std::atomic<uint64_t> q_values{0};   // the fence values handed out (a submit or a present takes one each)
std::mutex replay_mu;   // one replay at a time (the game submits from several threads)
// (RX_VK_BATCH: see queue_flush_locked)
VkSemaphore batch_tl = VK_NULL_HANDLE;
uint64_t batch_next = 0, batch_flushed = 0;
struct { std::vector<VkCommandBufferSubmitInfo> cbs; uint64_t signal = 0; bool any_signal = false; uint64_t rec = 0; } pend;   // (rec: the recordings' timeline value the held ones reach)
int64_t st_flushes, st_batched_cbs;
enum { FR_OTHER, FR_POLL, FR_WAIT, FR_SLOT, FR_PRESENT, FR_CAP, FR_VALUE, FR_COUNT };
int64_t st_reason[FR_COUNT];
int flush_reason = FR_OTHER;
std::chrono::steady_clock::time_point pend_since;

struct Recording {
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandPool upload_pool = VK_NULL_HANDLE;   // (its own: a parallel replay's threads record uploads into it while the
                                                  // replay thread records cmd, and a pool is one thread's at a time)
    VkCommandBuffer cmd = VK_NULL_HANDLE, upload = VK_NULL_HANDLE;
    // (a parallel replay's segment: new images' first layout changes, submitted before its uploads. Its own pool: any
    // segment's worker records into the earliest open segment's, while that segment's worker records cmd and upload)
    VkCommandPool init_pool = VK_NULL_HANDLE;
    VkCommandBuffer init = VK_NULL_HANDLE;
    uint64_t value = 0;   // the timeline value its submit signals (reusable once reached)
    int slot = 0;         // its index: the draw path keeps a descriptor pool and stream per recording
    // (its slot comes round again once its own submit is done: a fence of its own. The timeline is signalled as far
    // as no older submission is unfinished (submit_job), and a slot waiting on its value could wait on the replay)
    VkFence fence = VK_NULL_HANDLE;
    bool submitted = false;
    uint64_t batch = 0;   // (RX_VK_BATCH: the batch it went out in, waited for on batch_tl; 0: its own fence)
};
// (24: the event waits cut a frame's command buffers into pieces, each with recordings of its own, and with 16 the
// replay waited ~1.5 ms a frame for a slot the GPU still had)
constexpr int kRecordings = 48;   // (test: 24 before; with RX_VK_BATCH the ring came round to held submits once a frame)
Recording recordings[kRecordings];
int recording_index = 0;

/// (a recording slot's last submit done: before it is used again)
void recording_wait(Recording& r) {
    if (!r.submitted) return;
    if (r.batch) {
        {
            std::lock_guard lk(dev.present_mu);
            if (batch_flushed < r.batch) flush_reason = FR_SLOT, queue_flush_locked();
        }
        VkSemaphoreWaitInfo wi{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
        wi.semaphoreCount = 1;
        wi.pSemaphores = &batch_tl;
        wi.pValues = &r.batch;
        vkWaitSemaphores(dev.device, &wi, UINT64_MAX);
        r.batch = 0;
        r.submitted = false;
        return;
    }
    vkWaitForFences(dev.device, 1, &r.fence, VK_TRUE, UINT64_MAX);
    vkResetFences(dev.device, 1, &r.fence);
    r.submitted = false;
}

uint64_t completed() {
    uint64_t v = 0;
    vkGetSemaphoreCounterValue(dev.device, timeline, &v);
    return v;
}
// ---- (test) RX_VK_BATCH=1: the replay's submits held and sent as one vkQueueSubmit2 ----
// On amdgpu every command submission pays for all imported host memory (userptr) again; the replay made ~6 a frame.
// Held submits go out at the next point that needs the GPU's results or the queue's order: a wait, a readback, a
// present, a sparse bind, another submit (queue_flush_locked under dev.present_mu).
// (hookshot) on by default with Mesa's RADV, where it was measured (issue 413: a Radeon 8060S on Bazzite, ~10 -> 30-50 fps, the
// replay's queue submit ~70 -> ~6 ms a frame); elsewhere as before. RX_VK_BATCH=0 / 1: off / on on any driver
bool batch_on() {
    static const bool on = [] {
        bool radv = false;
        VkPhysicalDeviceDriverProperties dp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
        VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        p2.pNext = &dp;
        if (dev.physical) vkGetPhysicalDeviceProperties2(dev.physical, &p2), radv = dp.driverID == VK_DRIVER_ID_MESA_RADV;
        const char* e = getenv("RX_VK_BATCH");
        const bool v = e ? atoi(e) != 0 : radv;
        logf("vk: batched submits %s (RX_VK_BATCH%s)", v ? "on" : "off", !e && radv ? ", on with RADV" : "");
        return v;
    }();
    return on && batch_tl != VK_NULL_HANDLE;
}
} // namespace
void queue_flush_locked() {   // (dev.present_mu held)
    if (pend.cbs.empty()) return;
    VkSemaphoreSubmitInfo sig[3]{{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO}, {VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO},
                                 {VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO}};
    uint32_t n = 0;
    if (pend.any_signal) {
        sig[n].semaphore = timeline, sig[n].value = pend.signal, sig[n].stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        n++;
    }
    if (pend.rec) {   // (the recordings' timeline too: wait_recorded waits on it, and held submits carried no signal of it)
        sig[n].semaphore = rec_timeline, sig[n].value = pend.rec, sig[n].stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        n++;
    }
    sig[n].semaphore = batch_tl, sig[n].value = batch_next, sig[n].stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    n++;
    VkSubmitInfo2 si{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    si.commandBufferInfoCount = (uint32_t)pend.cbs.size();
    si.pCommandBufferInfos = pend.cbs.data();
    si.signalSemaphoreInfoCount = n;
    si.pSignalSemaphoreInfos = sig;
    PROF(PROF_QSUBMIT);
    const VkResult res = vkQueueSubmit2(dev.queue, 1, &si, VK_NULL_HANDLE);
    static int failures;
    if (res != VK_SUCCESS && failures++ < 8) logf("vk: batched submit %llu failed (%d)", (unsigned long long)batch_next, (int)res);
    if (res == VK_ERROR_DEVICE_LOST) device_lost();
    batch_flushed = batch_next;
    st_flushes++, st_batched_cbs += (int64_t)pend.cbs.size();
    st_reason[flush_reason]++;
    flush_reason = FR_OTHER;
    pend.cbs.clear(), pend.any_signal = false, pend.signal = 0, pend.rec = 0;
}
void queue_flush() {
    std::lock_guard lk(dev.present_mu);
    queue_flush_locked();
}
void device_wait_idle_locked() {   // (dev.present_mu held)
    queue_flush_locked();
    std::lock_guard pq(dev.present_queue_mu);
    vkDeviceWaitIdle(dev.device);
}
namespace {
void queue_flush_for(int why) {
    std::lock_guard lk(dev.present_mu);
    if (pend.cbs.empty()) return;
    flush_reason = why;
    queue_flush_locked();
}
// (RX_VK_BATCH_POLL_US=<us>: a progress poll sends held submits only once the oldest has waited that long; 0: always.
// Default 2000: the game polls many times a frame, and each poll that flushed was a whole command submission, the cost
// batching is there to avoid. RX 7800 XT, RADV, Kokiri: 525 -> 67 poll flushes per 120 presents, 38 -> 42.5 fps; issues
// 413, 528)
void queue_flush_poll() {
    static const int64_t us = getenv("RX_VK_BATCH_POLL_US") ? atoll(getenv("RX_VK_BATCH_POLL_US")) : 2000;
    std::lock_guard lk(dev.present_mu);
    if (pend.cbs.empty()) return;
    if (us > 0 && std::chrono::steady_clock::now() - pend_since < std::chrono::microseconds(us)) return;
    flush_reason = FR_POLL;
    queue_flush_locked();
}
} // namespace
namespace {
void wait_value(uint64_t v) {
    queue_flush_for(FR_VALUE);
    VkSemaphoreWaitInfo wi{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
    wi.semaphoreCount = 1;
    wi.pSemaphores = &timeline;
    wi.pValues = &v;
    vkWaitSemaphores(dev.device, &wi, UINT64_MAX);
}
} // namespace
uint64_t vk_timeline_completed() { return timeline ? completed() : 0; }
void vk_timeline_wait(uint64_t v) {   // (at most 250 ms: a submission held back behind the caller's thread goes on without it)
    if (!timeline || completed() >= v) return;
    queue_flush_for(FR_VALUE);
    VkSemaphoreWaitInfo wi{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
    wi.semaphoreCount = 1;
    wi.pSemaphores = &timeline;
    wi.pValues = &v;
    if (vkWaitSemaphores(dev.device, &wi, 250000000ull) == VK_TIMEOUT) {
        static std::atomic<int> logged{0};
        if (logged++ < 8) logf("vk: a capture's wait for write-back %llu timed out (done %llu)", (unsigned long long)v, (unsigned long long)completed());
    }
}
namespace {

int64_t st_wait_ns;   // (time the replay waited for a recording slot: the GPU behind)

// GPU time: a timestamp at the start and the end of each recording's command buffer, read when its slot comes
// round again (the work done by then)
VkQueryPool ts_pool = VK_NULL_HANDLE;
bool ts_pending[kRecordings];
int64_t st_gpu_ns, st_gpu_max_ns;
void gpu_time_collect(Recording& r) {
    if (!ts_pool || !ts_pending[r.slot]) return;
    ts_pending[r.slot] = false;
    uint64_t t[2] = {0, 0};
    if (vkGetQueryPoolResults(dev.device, ts_pool, 2 * r.slot, 2, sizeof t, t, 8, VK_QUERY_RESULT_64_BIT) != VK_SUCCESS)
        return;
    if (t[1] <= t[0]) return;
    const int64_t ns = (int64_t)((t[1] - t[0]) * (double)dev.props.limits.timestampPeriod);
    st_gpu_ns += ns;
    st_gpu_max_ns = std::max(st_gpu_max_ns, ns);
}
void gpu_time_begin(Recording& r) {
    if (!ts_pool) {
        VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qi.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qi.queryCount = 2 * kRecordings;
        if (vkCreateQueryPool(dev.device, &qi, nullptr, &ts_pool) != VK_SUCCESS) ts_pool = VK_NULL_HANDLE;
        if (!ts_pool) return;
    }
    vkCmdResetQueryPool(r.cmd, ts_pool, 2 * r.slot, 2);
    vkCmdWriteTimestamp2(r.cmd, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, ts_pool, 2 * r.slot);
}
void gpu_time_end(Recording& r) {
    if (!ts_pool) return;
    vkCmdWriteTimestamp2(r.cmd, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, ts_pool, 2 * r.slot + 1);
    ts_pending[r.slot] = true;
}

/// The next recording slot, begun, for the submission that will signal value
Recording& begin_recording(uint64_t value) {
    Recording& r = recordings[recording_index];
    recording_index = (recording_index + 1) % kRecordings;
    if (r.value) {
        PROF(PROF_SLOTWAIT);
        const auto t0 = std::chrono::steady_clock::now();
        recording_wait(r);
        st_wait_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
        gpu_time_collect(r);
        gprof_collect(r.slot);
    }
    vkResetCommandPool(dev.device, r.pool, 0);
    vkResetCommandPool(dev.device, r.upload_pool, 0);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(r.cmd, &bi);
    vkBeginCommandBuffer(r.upload, &bi);
    gpu_time_begin(r);
    gprof_begin_recording(r.slot, r.cmd, r.upload);
    r.value = value;
    memory_recording_value(value, completed());
    texture_begin_recording(r.upload, value);
    draw_begin_recording(r.slot, r.cmd, value);
    return r;
}
void submit_recording(Recording& r, bool signal_value = true, uint64_t signal_at = 0) {
    draw_flush_recording(r.slot);
    const bool uploads = texture_uploads_recorded();
    gprof_submitted(r.slot, uploads);
    gpu_time_end(r);
    vkEndCommandBuffer(r.upload);
    vkEndCommandBuffer(r.cmd);
    VkSemaphoreSubmitInfo signal{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    signal.semaphore = timeline;
    signal.value = signal_at ? signal_at : r.value;
    signal.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    VkCommandBufferSubmitInfo cb[2]{{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO}, {VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO}};
    uint32_t n = 0;
    if (uploads) cb[n++].commandBuffer = r.upload;
    cb[n++].commandBuffer = r.cmd;
    VkSubmitInfo2 si{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    si.commandBufferInfoCount = n;
    si.pCommandBufferInfos = cb;
    VkSemaphoreSubmitInfo signals[2] = {signal, {VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO}};
    uint32_t nsig = signal_value ? 1 : 0;
    uint64_t seq = 0;
    if (rec_timeline) {   // (and its number on the recordings' timeline; every caller holds replay_mu: in order)
        seq = ++rec_seq;
        signals[nsig].semaphore = rec_timeline;
        signals[nsig].value = seq;
        signals[nsig].stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        nsig++;
        std::lock_guard mk(rec_map_mu);
        for (uint64_t v : rec_open_values) value_rec.emplace(v, seq);
        if (r.value) value_rec.emplace(r.value, seq);
        rec_open_values.clear();
        while (value_rec.size() > 8192) value_rec.erase(value_rec.begin());
    }
    si.signalSemaphoreInfoCount = nsig;
    si.pSignalSemaphoreInfos = nsig ? signals : nullptr;
    std::lock_guard lk(dev.present_mu);   // (one queue: submits and presents take turns)
    if (batch_on()) {
        if (pend.cbs.empty()) batch_next++, pend_since = std::chrono::steady_clock::now();
        for (uint32_t i = 0; i < n; i++) pend.cbs.push_back(cb[i]);
        if (signal_value) pend.signal = std::max(pend.signal, signal.value), pend.any_signal = true;
        if (seq) pend.rec = std::max(pend.rec, seq);
        r.batch = batch_next;
        r.submitted = true;
        // (RX_VK_BATCH_MAX=<command buffers>: held at most this many, default 32)
        static const size_t cap = getenv("RX_VK_BATCH_MAX") ? (size_t)std::max(1, atoi(getenv("RX_VK_BATCH_MAX"))) : 32;
        if (pend.cbs.size() >= cap) flush_reason = FR_CAP, queue_flush_locked();
        return;
    }
    PROF(PROF_QSUBMIT);
    const VkResult res = vkQueueSubmit2(dev.queue, 1, &si, r.fence);
    r.submitted = res == VK_SUCCESS;
    static int failures;
    if (res != VK_SUCCESS && failures++ < 8) logf("vk: submit %llu failed (%d)", (unsigned long long)r.value, (int)res);
    if (res == VK_ERROR_DEVICE_LOST) device_lost();
}

// ---- NVN state (the draw path's, vk_state.h) ----
NvnState state;
// (each of the game's queues has one of its own, as the console's channels do: their command buffers are replayed in
// turns at their event waits (submit_job), and one's draws went on with what another's had set. A queue seen first
// starts from the state the replay was at, as all of them did before. The queue-less replays (RX_VK_ASYNC=0, the
// bench) keep the one above. RX_VK_QUEUE_STATE=0: one for all of them, as before.)
std::unordered_map<uint64_t, NvnState> qstates;
NvnState* cur_state = &state;
bool queue_state_on() {
    static const bool v = !getenv("RX_VK_QUEUE_STATE") || atoi(getenv("RX_VK_QUEUE_STATE")) != 0;
    return v;
}
/// The state the replay goes on with: its queue's. (The draw path's caches made afresh: they are keyed on the state's
/// counters, which the queues' states share no order of.)
void use_queue_state(uint64_t queue) {
    NvnState* s = &state;
    if (queue && queue_state_on()) {
        auto it = qstates.find(queue);
        if (it == qstates.end()) it = qstates.emplace(queue, *cur_state).first;
        s = &it->second;
    }
    if (s == cur_state) return;
    cur_state = s;
    static uint64_t gen_base = 0;
    gen_base += 1ull << 32;
    s->bind_gen = gen_base, s->rt_serial = gen_base;
    for (int k = 0; k < kNvnStages; k++) s->tex_gen[k] = gen_base, s->ssbo_gen[k] = gen_base;
}

template <class T>
T payload(const uint8_t* a, uint32_t bytes) {
    T v{};
    std::memcpy(&v, a, std::min<size_t>(sizeof v, bytes));
    return v;
}

int64_t st_barriers, st_ub_updates, st_ub_overlays, st_submits;
std::map<int32_t, int64_t> st_barrier_flags;   // (the game's nvnCommandBufferBarrier bits: how many of each combination, the frame log)
std::mutex st_barrier_flags_mu;   // (a parallel replay's threads count their barriers too)
// ---- events in replay order: the values the stream's signals and the CPU's set, for the GPU's waits on them. The
// game's three queues order their work with events (a command buffer waits for another queue's to reach a stage);
// replayed on one queue in the order they were submitted, a waiter submitted before its signaller ran first ----
std::unordered_map<uint64_t, uint32_t> ev_values;
// (hookshot) The CPU's signals, while a replay holds replay_mu: the main thread stopped for it in nvnEventSignal, ~1.7 ms
// a frame of the critical path. They wait here for the replay's next look at the events (ev_cpu_apply).
std::mutex ev_cpu_mu;
std::vector<std::pair<uint64_t, uint32_t>> ev_cpu_pending;
std::atomic<bool> ev_cpu_any{false};
std::unordered_set<uint64_t> cpu_events;   // (events the CPU has signalled, nvnEventSignal; ev_cpu_mu)
void ev_cpu_apply() {
    if (!ev_cpu_any.load(std::memory_order_acquire)) return;
    std::lock_guard lk(ev_cpu_mu);
    for (const auto& e : ev_cpu_pending) ev_values[e.first] = e.second, cpu_events.insert(e.first);
    ev_cpu_pending.clear();
    ev_cpu_any.store(false, std::memory_order_release);
}
int64_t st_waits_met, st_waits_unmet;
bool event_wait_met(uint32_t have, uint32_t mode, uint32_t want) {
    return mode == 0 ? have == want : (int32_t)(have - want) >= 0;   // (NVN: EQUAL, GEQUAL_WRAP)
}
using EvMap = std::unordered_map<uint64_t, uint32_t>;
uint32_t event_value_in(const EvMap& ev, uint64_t gpu) {   // (one the stream never set: as the CPU has it)
    auto it = ev.find(gpu);
    if (it != ev.end()) return it->second;
    uint32_t v = 0;
    host->gpu_read(gpu, &v, 4);
    return v;
}
thread_local bool suspend_ok = false;               // (a serial replay that stops at a wait not met: submit_job)
thread_local const uint8_t* suspend_at = nullptr;   // (the wait it stopped at)
uint64_t ts_anchor_ns = 0;   // (the last present's time: counter timestamps are scaled from it)

/// A GPU-timeline write of guest memory (the stream's own writes: they reach memory in command order)
void gpu_write(VkCommandBuffer cmd, uint64_t gpu, const void* data, uint64_t n) {
    const uint8_t* d = (const uint8_t*)data;
    while (n) {   // (vkCmdUpdateBuffer: up to 64 KB, 4-byte aligned)
        const uint64_t k = std::min<uint64_t>(n, 65536);
        BufferRef b = gpu_buffer_bind(gpu, k, true);
        if (!b || (b.offset & 3) || (k & 3)) {   // (unaligned or outside the pools: on the CPU, as before)
            guest_cpu_write(gpu, d, k);
        } else {
            vkCmdUpdateBuffer(cmd, b.buffer, b.offset, k, d);
        }
        gpu += k, d += k, n -= k;
    }
}

// ---- uniform updates (UPDATE_UB): written at the end of their recording; ub_overlay applies them to copies ----
struct UbUpdate {
    uint64_t value;   // the submission that writes it
    uint64_t dst;
    std::vector<uint8_t> bytes;
};
std::deque<UbUpdate> ub_updates;   // in stream order, oldest first
std::atomic<uint64_t> ub_list_epoch{1};   // (bumped when that deque gains or loses an update)
thread_local uint64_t ub_gen = 0;   // (each recording thread's: its draws' overlays change with what it passed)

// ---- parallel replay (RX_VK_PARALLEL=<threads>): MKW hands over nearly a frame's draws in one command buffer with
// its present, and waits for the last frame's GPU work before it submits the next: the replay of that command buffer
// (6-10 ms) was on the frame's critical path. A pass over its ops on the replay thread keeps only the state, and
// what has to happen in stream order (counter reports and the render conditions that read them, uniform updates);
// segments of it are then recorded on several threads from the state there, each into a command buffer of its own,
// submitted in order. ----
enum class Mode { Serial, Dry, Worker };
thread_local Mode rmode = Mode::Serial;
thread_local size_t ub_limit = SIZE_MAX;   // (a worker: the uniform updates before its place in the stream)
thread_local size_t cond_at = 0;           // (a worker: the next render condition's result)
std::vector<uint8_t> par_conds;            // (the render conditions' results, in stream order: the dry pass's)
struct ParPoint {
    const uint8_t* p;
    NvnState s;
    size_t ub, cond;
    int64_t work;
};
std::vector<ParPoint> par_points;   // (every 64 draws and dispatches of the dry pass)
bool dry_clears_on() {   // (the dry pass marks a clear's targets too: NOP_CLEAR_COLOR in replay())
    static const bool v = getenv("RX_VK_NO_DRY_CLEARS") == nullptr;
    return v;
}
int64_t dry_work = 0;
size_t ub_first_of_recording = 0;  // (the current recording's first entry)
uint64_t rec_value_now = 0;

bool deferred_late() {   // (RX_VK_DEFERRED_LATE=1: deferred binds read from the game's arrays at the replay)
    static const bool v = getenv("RX_VK_DEFERRED_LATE") != nullptr;
    return v;
}

void ub_prune(uint64_t done) {
    if (ub_updates.empty() || ub_updates.front().value > done) return;
    while (!ub_updates.empty() && ub_updates.front().value <= done) {
        ub_updates.pop_front();
        if (ub_first_of_recording) ub_first_of_recording--;
    }
    ub_list_epoch.fetch_add(1, std::memory_order_relaxed);
}

} // namespace

uint64_t ub_generation() {
    return ub_gen;
}
void ub_thread_off() {
    ub_limit = 0;
}
/// The open gap around [addr, addr+bytes) that no update covers. False when one does. [lo, hi) is that gap.
bool ub_gap(const uint64_t* dst, const uint64_t* end, int n, uint64_t addr, uint64_t bytes, uint64_t* lo, uint64_t* hi) {
    if (!bytes || addr + bytes < addr) return false;
    uint64_t left = 0, right = ~uint64_t{0};
    const uint64_t stop = addr + bytes;
    for (int i = 0; i < n; i++) {
        if (dst[i] < stop && addr < end[i]) return false;
        if (end[i] <= addr && end[i] > left) left = end[i];
        if (dst[i] >= stop && dst[i] < right) right = dst[i];
    }
    *lo = left;
    *hi = right;
    return true;
}
struct UbGapCache {
    uint64_t epoch = 0, lo = 0, hi = 0;
    size_t first = 0, lim = 0;
    bool on = false;
};
thread_local UbGapCache ub_gap_cache;
bool ub_gap_holds(uint64_t addr, uint64_t n, size_t first, size_t lim) {
    return n && ub_gap_cache.on && ub_gap_cache.epoch == ub_list_epoch.load(std::memory_order_relaxed) &&
           ub_gap_cache.first == first && ub_gap_cache.lim == lim && addr >= ub_gap_cache.lo &&
           addr + n >= addr && addr + n <= ub_gap_cache.hi;
}
bool ub_try_gap(size_t first, size_t lim, uint64_t addr, uint64_t n) {
    if (first > lim) return false;
    const uint64_t epoch = ub_list_epoch.load(std::memory_order_relaxed);
    static thread_local std::vector<uint64_t> dst, end;
    dst.clear();
    end.clear();
    dst.reserve(lim - first);
    end.reserve(lim - first);
    for (size_t i = first; i < lim; i++) {
        dst.push_back(ub_updates[i].dst);
        end.push_back(ub_updates[i].dst + ub_updates[i].bytes.size());
    }
    uint64_t lo = 0, hi = 0;
    if (!ub_gap(dst.data(), end.data(), (int)dst.size(), addr, n, &lo, &hi)) return false;
    if (ub_list_epoch.load(std::memory_order_relaxed) != epoch) return false;   // (the list changed: do not keep the gap)
    ub_gap_cache = {epoch, lo, hi, first, lim, true};
    return true;
}
// (a draw reads the same uniform words as the draw before it. The gap is what says later updates missed the word:
// it is kept while that gap covers it and its page is unstamped)
struct UbWord {
    uint64_t addr = 0, rec = 0;
    uint32_t page = 0, v = 0;
    bool on = false;
};
thread_local UbWord ub_words[128];
bool ub_word_hit(uint64_t addr, size_t lim, uint32_t* out) {
    UbWord& w = ub_words[(addr >> 2) & 127];
    if (!w.on || w.addr != addr || w.rec != rec_value_now || w.page != memory_page_stamp(addr)) return false;
    if (!ub_gap_holds(addr, 4, 0, lim)) return false;
    *out = w.v;
    return true;
}
// (a range an update does overlap: the stream copy stays while the gap's update list is unchanged and the copy is
// still in the recording's stream. A range in the gap has no copy)
struct UbCopy {
    const uint8_t* base = nullptr;
    uint64_t addr = 0, n = 0, at = 0, rec = 0, view = 0, epoch = 0;
    size_t lim = 0, first = 0;
    bool on = false;
};
thread_local UbCopy ub_copies[8];
bool ub_copy_hit(uint64_t addr, uint64_t n, const uint8_t* base, size_t first, size_t lim, uint64_t* at) {
    UbCopy& c = ub_copies[((addr >> 4) ^ n) & 7];
    if (!c.on || c.base != base || c.addr != addr || c.n != n || c.first != first || c.lim != lim ||
        c.rec != rec_value_now || c.epoch != ub_list_epoch.load(std::memory_order_relaxed) ||
        c.view != memory_capture_view())
        return false;
    if (!stream_holds(base, c.at, n)) return false;
    *at = c.at;
    return true;
}
void ub_copy_keep(uint64_t addr, uint64_t n, const uint8_t* base, size_t first, size_t lim, uint64_t at,
                  uint64_t view, uint64_t epoch) {
    if (view != memory_capture_view()) return;
    if (epoch != ub_list_epoch.load(std::memory_order_relaxed)) return;
    UbCopy& c = ub_copies[((addr >> 4) ^ n) & 7];
    c = {};
    c.on = true, c.base = base, c.addr = addr, c.n = n, c.at = at, c.first = first, c.lim = lim;
    c.rec = rec_value_now, c.view = view, c.epoch = epoch;
}
uint64_t ub_overlay(uint64_t addr, uint64_t n, uint8_t* base, const std::function<uint64_t(uint64_t)>& alloc) {
    // (a worker: those before its place in the stream; a background thread: none, not even the list's size)
    const size_t lim = ub_limit ? std::min(ub_limit, ub_updates.size()) : 0;
    static const bool cur_only = getenv("RX_VK_UB_CUR_ONLY") != nullptr;   // (A/B: this submission's updates only)
    const size_t first = cur_only ? std::min(ub_first_of_recording, lim) : 0;
    if (ub_gap_holds(addr, n, first, lim)) return UINT64_MAX;
    const uint64_t view = memory_capture_view();
    uint64_t kept = 0;
    if (ub_copy_hit(addr, n, base, first, lim, &kept)) return kept;
    if (ub_try_gap(first, lim, addr, n)) return UINT64_MAX;
    const uint64_t epoch = ub_list_epoch.load(std::memory_order_relaxed);
    bool any = false;
    for (size_t i = first; i < lim; i++) {
        const UbUpdate& u = ub_updates[i];
        if (u.dst < addr + n && addr < u.dst + u.bytes.size()) {
            any = true;
            break;
        }
    }
    if (!any) return UINT64_MAX;
    const uint64_t at = alloc(n);
    if (at == UINT64_MAX) return UINT64_MAX;
    uint8_t* p = base + at;
    if (!memory_read_as_submitted(addr, p, n)) std::memset(p, 0, n);
    for (size_t i = first; i < lim; i++) {
        const UbUpdate& u = ub_updates[i];
        const uint64_t lo = std::max(addr, u.dst), hi = std::min(addr + n, u.dst + u.bytes.size());
        if (lo < hi) std::memcpy(p + (lo - addr), u.bytes.data() + (lo - u.dst), hi - lo);
    }
    st_ub_overlays++;
    ub_copy_keep(addr, n, base, first, lim, at, view, epoch);
    return at;
}

/// Guest memory as the stream's place sees it: as it was at the submission (the capture), with the uniform updates
/// the stream made before this place (a worker: before its segment) over it. For what the replay reads on the CPU for
/// the GPU's sake (handles in uniform and storage buffers): the game's memory now may be its next frames'.
bool ub_read(uint64_t addr, void* dst, uint64_t n) {
    const size_t lim = ub_limit ? std::min(ub_limit, ub_updates.size()) : 0;
    if (n == 4) {
        uint32_t v = 0;
        if (ub_word_hit(addr, lim, &v)) {
            std::memcpy(dst, &v, 4);
            return true;
        }
    }
    const uint32_t page = n == 4 ? memory_page_stamp(addr) : 0;
    if (!memory_read_as_submitted(addr, dst, n)) return false;
    // (most words are not inside an inline update: the scan that finds that is kept for the gap, and the next words
    // in the gap do not walk the list. A word kept from that gap is the capture's own bytes)
    if (ub_gap_holds(addr, n, 0, lim) || ub_try_gap(0, lim, addr, n)) {
        if (n == 4 && page == memory_page_stamp(addr)) {
            UbWord& w = ub_words[(addr >> 2) & 127];
            uint32_t v = 0;
            std::memcpy(&v, dst, 4);
            w = {};
            w.on = true, w.addr = addr, w.v = v, w.rec = rec_value_now, w.page = page;
        }
        return true;
    }
    for (size_t i = 0; i < lim; i++) {
        const UbUpdate& u = ub_updates[i];
        const uint64_t lo = std::max(addr, u.dst), hi = std::min(addr + n, u.dst + u.bytes.size());
        if (lo < hi) std::memcpy((uint8_t*)dst + (lo - addr), u.bytes.data() + (lo - u.dst), hi - lo);
    }
    return true;
}

namespace {

/// The recording's uniform updates into guest memory (after its draws: the ones that came after an update read
/// their overlay copies)
void flush_ub_updates(VkCommandBuffer cmd) {
    if (ub_first_of_recording >= ub_updates.size()) return;
    GprofSpan gp(cmd, "ubupd (the stream's uniform updates -> guest memory, with its barriers)");
    gp.items = (int)(ub_updates.size() - ub_first_of_recording);
    full_barrier(cmd);
    for (size_t i = ub_first_of_recording; i < ub_updates.size(); i++)
        gpu_write(cmd, ub_updates[i].dst, ub_updates[i].bytes.data(), ub_updates[i].bytes.size()), gp.bytes += ub_updates[i].bytes.size();
    full_barrier(cmd);
}

/// (mesh shading: the stages its resources are bound to, beyond NVN's six)
void log_stage_bind(uint32_t op, int stage) {
    static int logged[64];
    if (stage >= 0 && stage < 64 && logged[stage]++ < 3) logf("vk: record %u binds to stage %d", op, stage);
}

void maybe_split(VkCommandBuffer& cmd);

/// The handles already stored, so a rebind of the same ones does not rebuild descriptors
bool handles_unchanged(const uint64_t* have, const uint64_t* next, int n) {
    if (n < 0) return false;
    for (int i = 0; i < n; i++)
        if (have[i] != next[i]) return false;
    return true;
}
/// A uniform or storage binding already stored
bool buf_unchanged(int32_t stage, int32_t index, uint64_t addr, uint64_t size, int32_t nstage, int32_t nindex,
                   uint64_t naddr, uint64_t nsize) {
    return stage == nstage && index == nindex && addr == naddr && size == nsize;
}

void replay(VkCommandBuffer& cmd, NvnState& s, const uint8_t* p, const uint8_t* end = nullptr) {
    const Mode mode = rmode;
    const bool dry = mode == Mode::Dry, worker = mode == Mode::Worker;
    for (;;) {
        if (p == end) return;
        nvn_op_hdr h;
        std::memcpy(&h, p, sizeof h);
        const uint8_t* a = p + 8;
        const uint32_t avail = h.size - 8;
        switch (h.op) {
        case NOP_END: return;
        case NOP_BEGIN: case NOP_JUMP: case NOP_CALL: break;
        // ---- memory the stream writes: on the GPU timeline ----
        case NOP_SIGNAL_EVENT: {
            if (!worker) {   // (in stream order: the replay thread's pass)
                const auto e = payload<nvn_p_signal>(a, avail);
                ev_cpu_apply();
                auto it = ev_values.find(e.gpu);
                uint32_t old = it != ev_values.end() ? it->second : 0;
                if (it == ev_values.end()) host->gpu_read(e.gpu, &old, 4);
                ev_values[e.gpu] = e.mode == NVN_EVENT_SIGNAL_MODE_ADD ? old + e.value : e.value;
            }
            if (dry) break;
            const auto e = payload<nvn_p_signal>(a, avail);
            uint32_t v = e.value;
            if (e.mode == NVN_EVENT_SIGNAL_MODE_ADD) {   // (the add on the CPU's view: events are few and in order)
                uint32_t old = 0;
                host->gpu_read(e.gpu, &old, 4);
                v = old + e.value;
            }
            end_pass(cmd);
            GprofSpan gp(cmd, "event signal (barrier, write, barrier)");
            full_barrier(cmd);
            gpu_write(cmd, e.gpu, &v, 4);
            break;
        }
        case NOP_WAIT_EVENT: {   // (a wait for an event another queue's work signals: submit_job orders them)
            if (worker) break;
            const auto w = payload<nvn_p_wait_event>(a, avail);
            ev_cpu_apply();
            const uint32_t have = event_value_in(ev_values, w.gpu);
            if (event_wait_met(have, w.mode, w.value)) {
                st_waits_met++;
            } else if (suspend_ok) {   // (on once that work is replayed; a dry pass: its part up to here)
                suspend_at = p;
                return;
            } else {
                st_waits_unmet++;
                static int logged;
                if (logged++ < 20 || st_waits_unmet % 5000 == 0)
                    logf("vk: a wait not met in replay order: event %#llx wants %s %u, has %u (submit %llu; %lld met, %lld "
                         "not)", (unsigned long long)w.gpu, w.mode == 0 ? "==" : ">=", w.value, have,
                         (unsigned long long)st_submits, (long long)st_waits_met, (long long)st_waits_unmet);
            }
            break;
        }
        case NOP_REPORT_COUNTER: {
            if (worker) break;   // (written by the dry pass, in stream order)
            const auto c = payload<nvn_p_counter>(a, avail);
            const bool flag_written = (c.type & NVN_COUNTER_WRITTEN) != 0;   // (a clear flag: nvn_native.c wrote it as it was recorded)
            const bool samples = (uint32_t)c.type == NVN_COUNTER_TYPE_SAMPLES_PASSED;
            // (timestamps on the guest tick clock in counter ticks (19.2 MHz, nvnDeviceGetTimestampInNanoseconds
            // converts), the replay's time; occlusion counts grow with every draw, as in the WebGPU renderer)
            // (the GPU's time as the game measures it: from the frame's present on, at RX_VK_TS_SCALE (default 0.1)
            // times the real clock. The reports are written as the replay records them, so the real clock gave the
            // replay's time and its lag, not the GPU's: MKW's dynamic resolution moved every frame (each new size a
            // new set of ~20 render targets, re-registered textures, 20-25 ms of replay a frame) and kept low. The
            // GPU takes 6-7 ms a frame at the resolution it then holds. RX_VK_TS_SCALE=1: the real clock;
            // RX_VK_TS_GLOBAL=1: scaled from the start, not from each present.)
            static const double ts_scale = getenv("RX_VK_TS_SCALE") ? atof(getenv("RX_VK_TS_SCALE")) : 0.1;
            static const bool ts_global = getenv("RX_VK_TS_GLOBAL") != nullptr;
            static uint64_t ts_base = 0;
            uint64_t ns = host->ticks() * 32;
            if (ts_scale != 1.0) {
                const uint64_t base = ts_global ? (ts_base ? ts_base : (ts_base = ns)) : std::min(ns, ts_anchor_ns ? ts_anchor_ns : ns);
                ns = base + (uint64_t)((double)(ns - base) * ts_scale);
            }
            // (hookshot) Every samples-passed report reads 4096: everything visible. The count of 16 a draw since the reset went
            // up and down from frame to frame, and OoT's sun visibility followed it: for one frame in ~70 the sunlight left the
            // ground and the haze brightened over the whole picture (one-frame flicker, most of it at load-in; 30-52 per 2,700
            // frames in Kokiri, 4-6 with this, RX 7900 GRE). RX_VK_OCC_FIXED=<n>: that value; -1: the per-draw count as before
            static const int64_t occ_fixed = getenv("RX_VK_OCC_FIXED") ? atoll(getenv("RX_VK_OCC_FIXED")) : 4096;
            const uint64_t rec[2] = {samples ? (occ_fixed >= 0 ? (uint64_t)occ_fixed : s.samples_passed) : 0, ns * 12 / 625};
            if (dump_on())
                logf("vk: dump-op %03d REPORT_COUNTER type %d at %#llx: %llu", dump_seq(), (int)c.type,
                     (unsigned long long)c.addr, (unsigned long long)rec[0]);
            // (written now, in stream order, not on the GPU timeline: SetRenderEnableConditional reads these words at
            // replay time, and MKW gates clears with them as TOTK does (the CPU writes 1 to ask for a clear, the report
            // after the clear resets it). A GPU-timeline write left the 1 in place for the stream's later conditions on
            // the same words, depending on how far the GPU was: extra clears of the HDR target, black frames at random;
            // and its late write could undo the game's next request. No pass ends for it either.)
            if (flag_written) guest_cpu_write(c.addr + 8, rec + 1, 8);
            else guest_cpu_write(c.addr, rec, sizeof rec);
            memory_cpu_owned(host->gpu_guest(c.addr), sizeof rec);   // (vk_memory.cpp: kept out of the mirrors' write-backs)
            break;
        }
        case NOP_RESET_COUNTER: {
            const auto c = payload<nvn_p_counter>(a, avail);
            if (c.type == NVN_COUNTER_TYPE_SAMPLES_PASSED) s.samples_passed = 0;
            break;
        }
        case NOP_COPY_B2B: {
            if (dry) break;
            ParLock lk;
            const auto c = payload<nvn_p_copy_bb>(a, avail);
            BufferRef src = gpu_buffer_bind(c.src, c.size, false), dst = gpu_buffer_bind(c.dst, c.size, true);
            {   // (RX_VK_WATCH_RANGE=<lo>:<hi>: buffer copies into it or out of it, 400 lines)
                static const std::pair<uint64_t, uint64_t> range = [] {
                    unsigned long long lo = 0, hi = 0;
                    if (const char* e = getenv("RX_VK_WATCH_RANGE")) sscanf(e, "%llx:%llx", &lo, &hi);
                    return std::pair<uint64_t, uint64_t>{lo, hi};
                }();
                static int logged;
                const bool hit = range.second && ((c.dst < range.second && c.dst + c.size > range.first) ||
                                                  (c.src < range.second && c.src + c.size > range.first));
                if (hit && logged++ < 400)
                    logf("vk: watch range buffer copy %#llx -> %#llx (%#llx bytes): source %s, destination %s", (unsigned long long)c.src,
                         (unsigned long long)c.dst, (unsigned long long)c.size, src ? "bound" : "NOT bound",
                         dst ? "bound" : "NOT bound (on the CPU)");
            }
            end_pass(cmd);
            GprofSpan gp(cmd, "copy buffer->buffer (with its barriers)");
            gp.bytes = c.size;
            full_barrier(cmd);
            if (src && dst) {
                const VkBufferCopy r{src.offset, dst.offset, c.size};
                vkCmdCopyBuffer(cmd, src.buffer, dst.buffer, 1, &r);
                const uint64_t guest = host->gpu_guest(c.dst);
                if (guest) memory_stamp_gpu(guest, c.size);
            } else {   // (outside the pools: on the CPU)
                for (uint64_t done = 0; done < c.size;) {
                    uint8_t buf[65536];
                    const uint64_t k = std::min<uint64_t>(sizeof buf, c.size - done);
                    if (!host->gpu_read(c.src + done, buf, k) || !guest_cpu_write(c.dst + done, buf, k)) break;
                    done += k;
                }
            }
            full_barrier(cmd);
            break;
        }
        case NOP_CLEAR_BUF: {   // (hookshot) nvnCommandBufferClearBuffer: on the GPU in order, as a buffer copy
            const auto c0 = payload<nvn_p_clear_buf>(a, avail);
            // (the small ones are the counters the reports and conditions use, which happen here on the CPU in stream order. Left
            // to the GPU, a clear earlier in the stream reached guest memory after a later report and wiped it, as far as the GPU
            // was behind: lights and whole rooms switched off by their conditions for a frame. RX_VK_CLEAR_CPU_MAX=0: on the GPU)
            static const uint64_t cpu_max = getenv("RX_VK_CLEAR_CPU_MAX") ? (uint64_t)atoll(getenv("RX_VK_CLEAR_CPU_MAX")) : 256;
            const bool in_stream = c0.size && c0.size <= cpu_max;
            if (in_stream && !worker) {
                uint32_t words[64];
                for (auto& w : words) w = c0.value;
                guest_cpu_write(c0.dst, words, std::min<uint64_t>(c0.size, sizeof words));   // (hookshot: not past words)
            }
            if (dry) break;
            ParLock lk;
            const auto c = c0;
            if (!c.size) break;
            BufferRef dst = gpu_buffer_bind(c.dst, c.size, true);
            {   // (EvilOOT: RX_VK_WATCH_RANGE, the clears into it too)
                unsigned long long lo = 0, hi = 0;
                static const char* e = getenv("RX_VK_WATCH_RANGE");
                if (e) sscanf(e, "%llx:%llx", &lo, &hi);
                static int logged;
                if (hi && c.dst < hi && c.dst + c.size > lo && logged++ < 400)
                    logf("vk: watch range buffer clear %#llx (%#llx bytes) to %#x: %s", (unsigned long long)c.dst,
                         (unsigned long long)c.size, (unsigned)c.value, dst ? "on the GPU" : "NOT bound");
            }
            end_pass(cmd);
            GprofSpan gp(cmd, "clear buffer (with its barriers)");
            gp.bytes = c.size;
            full_barrier(cmd);
            if (dst && !(dst.offset & 3) && !(c.size & 3)) {
                vkCmdFillBuffer(cmd, dst.buffer, dst.offset, c.size, c.value);
                const uint64_t guest = host->gpu_guest(c.dst);
                if (guest && !in_stream) memory_stamp_gpu(guest, c.size);
            } else if (!in_stream) {   // (outside the pools or unaligned: on the CPU)
                uint32_t buf[1024];
                for (auto& w : buf) w = c.value;
                for (uint64_t done = 0; done < c.size;) {
                    const uint64_t k = std::min<uint64_t>(sizeof buf, c.size - done);
                    if (!guest_cpu_write(c.dst + done, buf, k)) break;
                    done += k;
                }
            }
            full_barrier(cmd);
            break;
        }
        case NOP_COPY_B2T: if (!dry) { ParLock lk; copy_b2t(cmd, s, payload<nvn_p_copy_bt>(a, avail), true); } break;
        case NOP_COPY_T2B: if (!dry) { ParLock lk; copy_b2t(cmd, s, payload<nvn_p_copy_bt>(a, avail), false); } break;
        case NOP_COPY_T2T: if (!dry) { ParLock lk; copy_t2t(cmd, s, payload<nvn_p_copy_tt>(a, avail)); } break;
        case NOP_CLEAR_TEX: if (!dry) { ParLock lk; clear_tex(cmd, s, payload<nvn_p_clear_tex>(a, avail)); } break;
        case NOP_UPDATE_UB: {
            const auto u = payload<nvn_p_update_ub>(a, avail);
            if (worker) {   // (appended by the dry pass: the draws after it read it from here on)
                // (hookshot) counted only when the dry pass appended it (the same test as below): an update of no bytes, or of more
                // than the stream holds, is not in ub_updates, and counting it moved every later draw of the segment one update ahead
                if (u.size && u.size <= avail - sizeof u) ub_limit++, ub_gen++;
                break;
            }
            if (dump_on() && u.size && u.size <= avail - sizeof u) {   // (in a dump: each update, its first words)
                float f[12] = {};
                std::memcpy(f, a + sizeof u, std::min<size_t>(sizeof f, u.size));
                logf("vk: dump-op %03d UPDATE_UB %#llx +%#x: %.5g %.5g %.5g %.5g | %.5g %.5g %.5g %.5g | %.5g %.5g %.5g %.5g", dump_seq(),
                     (unsigned long long)u.dst, (unsigned)u.size, f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7], f[8], f[9], f[10], f[11]);
            }
            if (u.size && u.size <= avail - sizeof u) {
                UbUpdate up{rec_value_now, u.dst, std::vector<uint8_t>(a + sizeof u, a + sizeof u + u.size)};
                ub_updates.push_back(std::move(up));
                ub_list_epoch.fetch_add(1, std::memory_order_relaxed);
                ub_gen++;
                st_ub_updates++;
            }
            break;
        }
        // ---- state ----
        case NOP_SET_RT: s.rt = payload<nvn_p_set_rt>(a, avail), s.rt_serial++; break;
        case NOP_VIEWPORTS: {
            const auto v = payload<nvn_p_viewports>(a, avail);
            for (int i = 0; i < v.n && v.first + i < 16; i++) std::memcpy(s.viewport[v.first + i], v.v[i], 16);
            break;
        }
        case NOP_SCISSOR: {
            const auto c = payload<nvn_p_scissor>(a, avail);
            s.scissor[0] = c.x, s.scissor[1] = c.y, s.scissor[2] = c.w, s.scissor[3] = c.h;
            break;
        }
        case NOP_DEPTH_RANGE: {
            const auto d = payload<nvn_p_depth_range>(a, avail);
            s.depth_near = d.n, s.depth_far = d.f;
            break;
        }
        case NOP_BIND_PROGRAM: {
            const auto b = payload<nvn_p_bind_program>(a, avail);
            // (the same program bound again: its descriptors already stand)
            const bool same = b.prog && s.has_prog ? std::memcmp(&s.prog, b.prog, sizeof s.prog) == 0 : !b.prog && !s.has_prog;
            if (!same) s.bind_gen++;
            if (b.prog) s.prog = *b.prog;
            s.has_prog = b.prog != nullptr && b.prog->nstage > 0;
            break;
        }
        case NOP_BIND_STATE: {
            const auto b = payload<nvn_p_bind_state>(a, avail);
            switch (b.kind) {
            case 0: {
                nvn_blend_st st;
                std::memcpy(&st, b.bytes, sizeof st);
                s.blend[st.target & 7] = st;
                break;
            }
            case 1: std::memcpy(&s.color, b.bytes, sizeof s.color); break;
            case 2: std::memcpy(&s.chmask, b.bytes, sizeof s.chmask); break;
            case 3: std::memcpy(&s.ms, b.bytes, sizeof s.ms); break;
            case 4: std::memcpy(&s.poly, b.bytes, sizeof s.poly); break;
            case 5: std::memcpy(&s.ds, b.bytes, sizeof s.ds); break;
            default: break;
            }
            break;
        }
        case NOP_BIND_VATTR: {
            const auto v = payload<nvn_p_bind_vattr>(a, avail);
            std::memset(s.vattr, 0, sizeof s.vattr);
            for (int i = 0; i < v.n && i < 16; i++) s.vattr[i] = v.st[i];
            s.nvattr = std::min(v.n, 16);
            break;
        }
        case NOP_BIND_VSTREAM: {
            const auto v = payload<nvn_p_bind_vstream>(a, avail);
            for (int i = 0; i < v.n && i < 16; i++) s.vstream[i] = v.st[i];
            break;
        }
        case NOP_BIND_VBUF: {
            const auto v = payload<nvn_p_bind_vbuf>(a, avail);
            for (int i = 0; i < v.n && v.first + i < 16; i++) s.vbuf[v.first + i] = v.r[i];
            break;
        }
        case NOP_BIND_BUFS: {
            auto b = payload<nvn_p_bind_bufs>(a, avail);
            // (deferred binds: the ranges as the game's array held them at the submit, which the host copied into
            // the recording then (nvn_native.c resolve_deferred). Read again here, at the replay, the array was
            // often the next frame's already (MKW refills it as soon as the submit returns): the terrain patches
            // took other patches' placements (holes in the far terrain, the minimap's triangles). RX_VK_DEFERRED_LATE=1:
            // read at the replay, as before.)
            {   // (EvilOOT, RX_VK_LOG_DEFERRED=1: compute's deferred binds, as at the submit and as the array holds them now)
                static const bool on = getenv("RX_VK_LOG_DEFERRED") != nullptr;
                static std::atomic<int> told{0};
                if (on && b.src && b.stage == 5 && told < 400) {
                    NVNbufferRange now[16]{};
                    const int n = std::clamp(b.n, 0, 16);
                    host->gpu_read(b.src, now, (uint64_t)n * sizeof(NVNbufferRange));
                    for (int i = 0; i < n && told < 400; i++, told++)
                        logf("vk: deferred %s stage 5 slot %d (array %#llx): submit %#llx+%#llx, now %#llx+%#llx%s", b.ssbo ? "ssbo" : "ubo",
                             b.first + i, (unsigned long long)b.src, (unsigned long long)b.r[i].address, (unsigned long long)b.r[i].size,
                             (unsigned long long)now[i].address, (unsigned long long)now[i].size,
                             now[i].address != b.r[i].address || now[i].size != b.r[i].size ? "  DIFFERS" : "");
                }
            }
            if (b.src && deferred_late()) {
                const int n = std::clamp(b.n, 0, 16);
                std::memset(b.r, 0, sizeof b.r);
                host->gpu_read(b.src, b.r, (uint64_t)n * sizeof(NVNbufferRange));
            }
            for (int i = 0; i < b.n && i < 16; i++) {
                const int index = b.first + i;
                if (b.stage < 0 || b.stage >= kNvnStages || index < 0 || index >= 16) continue;
                {
                    static const bool on = getenv("RX_VK_LOG_MESH_BINDS") != nullptr;
                    static std::atomic<int> told{0};
                    if (on && b.stage >= 6 && b.r[i].size != 0x100 && told++ < 300)
                        logf("vk: bind (many) %s stage %d index %d addr %#llx size %#llx", b.ssbo ? "ssbo" : "ubo", b.stage, index,
                             (unsigned long long)b.r[i].address, (unsigned long long)b.r[i].size);
                }
                nvn_p_bind_buf& u = (b.ssbo ? s.ssbo : s.ubo)[b.stage][index];
                if (!buf_unchanged(u.stage, u.index, u.addr, u.size, b.stage, index, b.r[i].address, b.r[i].size)) {
                    s.bind_gen++;
                    if (b.ssbo) s.ssbo_gen[b.stage] = s.bind_gen;
                }
                u.stage = b.stage, u.index = index, u.addr = b.r[i].address, u.size = b.r[i].size;
            }
            break;
        }
        case NOP_BIND_UBO: case NOP_BIND_SSBO: {
            const auto b = payload<nvn_p_bind_buf>(a, avail);
            if (b.stage >= kNvnStages) log_stage_bind(h.op, b.stage);
            {   // (hookshot) RX_VK_LOG_MESH_BINDS=1: what the game binds to the task and mesh stages, 80 lines
                static const bool on = getenv("RX_VK_LOG_MESH_BINDS") != nullptr;
                static std::atomic<int> told{0};
                if (on && b.stage >= 6 && b.size != 0x100 && told++ < 300)
                    logf("vk: bind %s stage %d index %d addr %#llx size %#llx", h.op == NOP_BIND_UBO ? "ubo" : "ssbo", b.stage, b.index,
                         (unsigned long long)b.addr, (unsigned long long)b.size);
            }
            if (b.stage >= 0 && b.stage < kNvnStages && b.index >= 0 && b.index < 16) {
                nvn_p_bind_buf& u = (h.op == NOP_BIND_UBO ? s.ubo : s.ssbo)[b.stage][b.index];
                if (!buf_unchanged(u.stage, u.index, u.addr, u.size, b.stage, b.index, b.addr, b.size)) {
                    s.bind_gen++;
                    if (h.op == NOP_BIND_SSBO) s.ssbo_gen[b.stage] = s.bind_gen;
                }
                u = b;
            }
            break;
        }
        case NOP_BIND_TEX: case NOP_BIND_SAMP: case NOP_BIND_IMG: {
            nvn_p_bind_handles b;
            std::memset(&b, 0, sizeof b);
            std::memcpy(&b, a, std::min<size_t>(sizeof b, h.size - 8));
            if (b.src && deferred_late()) {   // (deferred: the handles as of the submit, as the ranges above)
                const int n = std::clamp(b.n, 0, 32);
                std::memset(b.h, 0, sizeof b.h);
                host->gpu_read(b.src, b.h, (uint64_t)n * 8);
            }
            if (b.stage >= kNvnStages) log_stage_bind(h.op, b.stage);
            if (b.stage < 0 || b.stage >= kNvnStages || b.first < 0) break;
            uint64_t* to = h.op == NOP_BIND_TEX ? s.tex[b.stage] : h.op == NOP_BIND_SAMP ? s.samp[b.stage] : s.img[b.stage];
            const int lim = h.op == NOP_BIND_IMG ? 32 : 128;
            int n = 0;
            while (n < b.n && b.first + n < lim) n++;
            if (!handles_unchanged(to + b.first, b.h, n)) {   // (the same handles: the draw keeps its descriptors)
                s.bind_gen++;
                s.tex_gen[b.stage] = s.bind_gen;
            }
            for (int k = 0; k < n; k++) to[b.first + k] = b.h[k];
            break;
        }
        case NOP_STENCIL: {
            const auto st = payload<nvn_p_stencil>(a, avail);
            for (int f = 0; f < 2; f++) {
                if (!(st.face & (1 << f))) continue;
                if (st.which == 0) s.stencil_value_mask[f] = (uint32_t)st.value_mask;
                else if (st.which == 1) s.stencil_mask[f] = (uint32_t)st.mask;
                else s.stencil_ref[f] = (uint32_t)st.ref;
            }
            break;
        }
        case NOP_POLY_OFFSET: {
            const auto o = payload<nvn_p_poly_offset>(a, avail);
            s.poly_offset[0] = o.factor, s.poly_offset[1] = o.units, s.poly_offset[2] = o.clamp;
            break;
        }
        case NOP_BLEND_COLOR: std::memcpy(s.blend_color, payload<nvn_p_blend_color>(a, avail).rgba, 16); break;
        case NOP_PRIMITIVE_RESTART: {
            const auto r = payload<nvn_p_restart>(a, avail);
            s.restart = r.enable != 0, s.restart_index = (uint32_t)r.index;
            break;
        }
        case NOP_DEPTH_CLAMP: s.depth_clamp = payload<nvn_p_flag>(a, avail).value != 0; break;
        case NOP_RASTER_DISCARD: s.raster_discard = payload<nvn_p_flag>(a, avail).value != 0; break;
        case NOP_RENDER_ENABLE:
            s.render_enabled = payload<nvn_p_flag>(a, avail).value != 0;
            if (dump_on()) logf("vk: dump-op %03d RENDER_ENABLE %d", dump_seq(), (int)s.render_enabled);
            break;
        case NOP_RENDER_COND: {   // (read on the CPU for now; VK_EXT_conditional_rendering later)
            if (worker) {   // (as the dry pass found it, with the reports before it written and none after)
                s.render_enabled = cond_at < par_conds.size() ? par_conds[cond_at] != 0 : true;
                cond_at++;
                break;
            }
            const auto r = payload<nvn_p_render_cond>(a, avail);
            uint64_t x = r.x, y = r.y;
            if (!r.snapped) {
                host->gpu_read(r.addr, &x, 8);
                host->gpu_read(r.addr + 16, &y, 8);
            }
            s.render_enabled = r.mode == 0 ? x == y : x != y;
            {   // (RX_VK_COND_ALWAYS=1 (test): every condition holds. The frame log's dark frames dump the last ones)
                static const bool always = getenv("RX_VK_COND_ALWAYS") != nullptr;
                const uint32_t at = cond_ring_at++ % (1 << 15);
                cond_ring[at] = {presents_now, r.addr, x, y, (int)r.mode, s.render_enabled};
                if (always) s.render_enabled = true;
            }
            if (dry) par_conds.push_back(s.render_enabled ? 1 : 0);
            if (dump_on())
                logf("vk: dump-op %03d RENDER_COND mode %d at %#llx: %llu vs %llu: %s", dump_seq(), r.mode,
                     (unsigned long long)r.addr, (unsigned long long)x, (unsigned long long)y,
                     s.render_enabled ? "on" : "off");
            break;
        }
        case NOP_SET_TEX_POOL: case NOP_SET_SAMP_POOL:
            s.bind_gen++;
            for (int k = 0; k < kNvnStages; k++) s.tex_gen[k] = s.bind_gen;
            break;
        case NOP_COPY_STRIDE: s.copy_stride = payload<nvn_p_copy_stride>(a, avail).stride; break;
        case NOP_PUSH_CONST: {   // (hookshot) compute push constants, kept in the state (vk_bank.cpp serves them from c0)
            const auto u = payload<nvn_p_push_const>(a, avail);
            if (u.size && u.size <= avail - sizeof u && (uint64_t)u.offset + u.size <= sizeof s.cs_push) {
                const uint8_t* src = a + sizeof u;
                const bool same = (uint64_t)u.offset + u.size <= s.cs_push_n && std::memcmp(s.cs_push + u.offset, src, u.size) == 0;
                std::memcpy(s.cs_push + u.offset, src, u.size);
                s.cs_push_n = std::max<uint32_t>(s.cs_push_n, u.offset + u.size);
                if (!same) s.bind_gen++;
            }
            break;
        }
        case NOP_BARRIER:
            if (dry) break;
            {
                const int32_t bits = payload<int32_t>(a, avail);
                std::lock_guard lk(st_barrier_flags_mu);
                st_barrier_flags[bits]++;
            }
            end_pass(cmd);
            full_barrier(cmd, "barrier (the game's nvnCommandBufferBarrier)");
            st_barriers++;
            break;
        // ---- work ----
        case NOP_CLEAR_COLOR:
            // (hookshot) in the dry pass a clear marks its pass's targets as the GPU's, as a draw does (dry_mark_written): the clear
            // opens a pass on the worker, and that marks them (begin_pass), but only when the worker gets there. A target the game
            // clears in one segment and samples in a later one was still "never written" for the later segment when it was
            // recorded first: it uploaded what the target's memory held, an upload that runs after the clear. 13 of them in 4,300
            // presents of Castle Town's market with the camera turning, each an 853 x 480 target made that frame ("written in
            // segment N, uploaded by the later segment N+2; written by: a pass's colour target"), among 38 one-frame flashes
            // against 9 on one thread. RX_VK_NO_DRY_CLEARS=1: as before
            if (s.render_enabled && dry && dry_clears_on()) dry_mark_written(s, false, true);
            if (s.render_enabled && !dry) {
                const auto c = payload<nvn_p_clear_color>(a, avail);
                clear_color_target(cmd, s, c.index, c.rgba, (uint32_t)c.mask);
            }
            break;
        case NOP_CLEAR_DS:
            if (s.render_enabled && dry && dry_clears_on()) dry_mark_written(s, false, true);   // (as NOP_CLEAR_COLOR above)
            if (s.render_enabled && !dry) {
                const auto c = payload<nvn_p_clear_ds>(a, avail);
                clear_depth_stencil(cmd, s, c.depth, c.depth_mask != 0, (uint32_t)c.stencil, (uint32_t)c.stencil_mask);
            }
            break;
        case NOP_DRAW: {
            const auto d = payload<nvn_p_draw>(a, avail);
            if (s.render_enabled && !s.raster_discard && !dry) {
                DrawArgs da;
                da.prim = d.prim, da.first = d.first, da.count = d.count;
                da.base_instance = d.base_instance, da.instances = d.instances;
                draw(cmd, s, da);
            }
            s.samples_passed += 16;
            break;
        }
        case NOP_DRAW_IDX: {
            const auto d = payload<nvn_p_draw_idx>(a, avail);
            if (s.render_enabled && !s.raster_discard && !dry) {
                DrawArgs da;
                da.prim = d.prim, da.indexed = true, da.itype = d.itype, da.count = d.count;
                da.base_vertex = d.base_vertex, da.base_instance = d.base_instance, da.instances = d.instances;
                da.ibuf = d.ibuf;
                draw(cmd, s, da);
            }
            s.samples_passed += 16;
            break;
        }
        case NOP_DRAW_INDIRECT: {
            const auto d = payload<nvn_p_draw_indirect>(a, avail);
            if (s.render_enabled && !s.raster_discard && !dry) {
                DrawArgs da;
                da.prim = d.prim, da.indexed = d.indexed != 0, da.itype = d.itype;
                da.indirect = d.indirect, da.count_addr = d.count_addr, da.max = d.max, da.stride = d.stride;
                da.ibuf = d.ibuf;
                draw(cmd, s, da);
            }
            s.samples_passed += 16;
            break;
        }
        case NOP_DRAW_MESH: {
            const auto d = payload<nvn_p_draw_mesh>(a, avail);
            static int logged_mesh;
            if (logged_mesh++ < 8)
                logf("vk: mesh draw kind %d: %d %d %d, indirect %#llx count %#llx max %d stride %d", d.kind, d.x, d.y,
                     d.z, (unsigned long long)d.indirect, (unsigned long long)d.count_addr, d.max, d.stride);
            if (s.render_enabled && !s.raster_discard && !dry) {
                DrawArgs da;
                da.mesh = true;
                // (DrawMeshTasks(first, count): the count of task workgroups, as Eden draws NVN2's mesh draws; a
                // first task other than 0 has no Vulkan counterpart)
                da.groups[0] = d.y, da.groups[1] = 1, da.groups[2] = 1;
                if (d.kind == 0) da.first_task = d.x;   // (hookshot: the task programs add it, vk_draw.cpp's mesh_args)
                da.indirect = d.indirect, da.count_addr = d.count_addr, da.max = d.max, da.stride = d.stride;
                draw(cmd, s, da);
            }
            s.samples_passed += 16;
            break;
        }
        case NOP_DISPATCH: {
            const auto d = payload<nvn_p_dispatch>(a, avail);
            if (s.render_enabled && !dry) dispatch(cmd, s, (uint32_t)d.x, (uint32_t)d.y, (uint32_t)d.z, 0);
            break;
        }
        case NOP_DISPATCH_INDIRECT: {
            const auto d = payload<nvn_p_dispatch_indirect>(a, avail);
            if (s.render_enabled && !dry) dispatch(cmd, s, 1, 1, 1, d.addr);
            break;
        }
        default: break;
        }
        if (h.op == NOP_DRAW || h.op == NOP_DRAW_IDX || h.op == NOP_DRAW_INDIRECT || h.op == NOP_DRAW_MESH ||
            h.op == NOP_DISPATCH || h.op == NOP_DISPATCH_INDIRECT) {
            if (dry) {
                const bool compute = h.op == NOP_DISPATCH || h.op == NOP_DISPATCH_INDIRECT;
                if (s.render_enabled && (compute || !s.raster_discard)) dry_mark_written(s, compute, dry_work == 0);
                if (++dry_work % 64 == 0) par_points.push_back({p + h.size, s, ub_updates.size(), par_conds.size(), dry_work});
            } else if (!worker) {
                maybe_split(cmd);
            }
        }
        p += h.size;
    }
}

// ---- chunks: a long submission goes to the GPU in parts (RX_VK_CHUNK=<draws and dispatches a part>, default 120;
// 0: whole), so the GPU starts on its first part while the rest is recorded. The game waits for its fences within
// the frame: with the whole submission recorded before the GPU saw any of it, its wait was the replay's time plus
// the GPU's. Only the last part signals the submission's timeline value; the parts' slots free with it. ----
Recording* rec_cur = nullptr;
uint64_t rec_cur_value = 0;
int rec_parts = 0;
int64_t part_work = 0;
// (a slot is used again once its own submit is done (recording_wait): a submission's parts may outnumber the slots,
// which a wait on the timeline value it signals at its end kept them from. A parallel replay's segments are recorded
// at once, kMaxBatch of them at most: fewer than the slots)
constexpr int kMaxParts = 32;
constexpr int kMaxBatch = 6;   // (three recording threads: more segments only took slots and submits)

void maybe_split(VkCommandBuffer& cmd) {
    static const int64_t every = getenv("RX_VK_CHUNK") ? atoll(getenv("RX_VK_CHUNK")) : 120;
    if (!rec_cur || every <= 0 || ++part_work < every || rec_parts + 1 >= kMaxParts || dump_on()) return;
    end_pass(cmd);
    flush_ub_updates(cmd);
    memory_end_submission(cmd);
    full_barrier(cmd, "barrier (recording end)");
    submit_recording(*rec_cur, false);
    ub_first_of_recording = ub_updates.size();
    Recording& r = begin_recording(rec_cur_value);
    memory_begin_submission();
    full_barrier(r.cmd, "barrier (recording start)");
    rec_cur = &r;
    cmd = r.cmd;
    part_work = 0;
    rec_parts++;
}

// ---- parallel replay: the threads and a command buffer's segments ----
// (RX_VK_PARALLEL=<threads recording a command buffer, the replay thread one of them>; 0, 1: off, the default. Segments
// recorded out of order left one-frame black scenes in play, 48-93 in a walk through Kakariko against none on the
// replay thread alone, at about the same frame rate here: 36-40 fps against 38-41)
// (hookshot) One cause of those is gone: the dry pass did not mark the targets of a clear (NOP_CLEAR_COLOR in replay()). With
// that, Castle Town's market and eight other places showed no black frame on 3 to 8 threads; the Kakariko walk was not run
// again, so it stays off. On a 9950X3D the replay's wall time in the market goes from 10.9 ms a frame to 7.75 on 3 threads.
int par_threads() {
    static const int n = std::clamp(getenv("RX_VK_PARALLEL") ? atoi(getenv("RX_VK_PARALLEL")) : 0, 0, 8);
    return n;
}
struct ParPool {
    std::mutex mu;
    std::condition_variable cv, done;
    std::deque<std::function<void()>> jobs;
    int pending = 0;
    bool started = false;
};
ParPool par_pool;
void par_worker_main() {
    SetThreadDescription(GetCurrentThread(), L"mkw_vk replay worker");
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
    for (;;) {
        std::function<void()> fn;
        {
            std::unique_lock lk(par_pool.mu);
            par_pool.cv.wait(lk, [] { return !par_pool.jobs.empty(); });
            fn = std::move(par_pool.jobs.front());
            par_pool.jobs.pop_front();
        }
        fn();
        std::lock_guard lk(par_pool.mu);
        if (--par_pool.pending == 0) par_pool.done.notify_all();
    }
}

/// A recording slot for a segment a worker records (its command buffer only: uploads go to the shared one)
Recording& begin_segment_recording(uint64_t value) {
    Recording& r = recordings[recording_index];
    recording_index = (recording_index + 1) % kRecordings;
    if (r.value) {
        PROF(PROF_SLOTWAIT);
        recording_wait(r);
        gpu_time_collect(r);
        gprof_collect(r.slot);
    }
    vkResetCommandPool(dev.device, r.pool, 0);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(r.cmd, &bi);
    r.value = value;
    return r;
}

struct Segment {
    const uint8_t *p, *end;
    const NvnState* s;
    size_t ub, cond;
};
/// One segment recorded into cmd from the state at its start (any thread: the replay thread's own included)
void record_segment(VkCommandBuffer& cmd, const Segment& g) {
    rmode = Mode::Worker;
    NvnState s = *g.s;
    ub_limit = g.ub, cond_at = g.cond;
    ub_gen++;
    full_barrier(cmd);   // (after what the command buffers before it did)
    replay(cmd, s, g.p, g.end);
    end_pass(cmd);
    full_barrier(cmd);
    rmode = Mode::Serial;
    ub_limit = SIZE_MAX;
}

/// The replay cannot go on: logged, the host's queued log written out (as hos_fatal does: a device lost's report is many
/// lines, and the queue went with the process), written to stderr and the process ended (exit code 3). For a parallel
/// replay's init command buffer that could not be ended: images already made and published lost their first layout
/// change, and any later submission may use them
[[noreturn]] void replay_fatal(const char* why) {
    logf("vk: FATAL: %s", why);
    if (host && host->log_flush) host->log_flush();
    std::fprintf(stderr, "vk: FATAL: %s\n", why);
    std::fflush(stderr);
    std::_Exit(3);
}
std::recursive_mutex par_mu;

/// A command buffer of the submission from h (its start, or where it stopped at a wait): on the replay thread alone
/// (RX_VK_PARALLEL off, a small one, dumps and profiles), or its dry pass there and its segments on the threads.
/// may_stop: up to a wait not met, which it returns (null: the command buffer done)
const uint8_t* replay_handle(VkCommandBuffer& cmd, const uint8_t* h, bool may_stop = false) {
    const int threads = par_threads();
    NvnState& state = *cur_state;
    suspend_at = nullptr;
    if (threads < 2 || !rec_cur || dump_on() || gprof_on() || probe_serial()) {
        suspend_ok = may_stop;
        replay(cmd, state, h);
        suspend_ok = false;
        return suspend_at;
    }
    // the dry pass: the state at every 64 draws and dispatches, the render conditions, the uniform updates
    const auto t_dry0 = std::chrono::steady_clock::now();
    const NvnState start = state;
    const size_t ub0 = ub_updates.size();
    par_points.clear(), par_conds.clear(), dry_work = 0;
    dry_marks_begin();
    rmode = Mode::Dry;
    VkCommandBuffer none = cmd;
    suspend_ok = may_stop;
    replay(none, state, h);
    suspend_ok = false;
    rmode = Mode::Serial;
    const uint8_t* const stop = suspend_at;   // (the segments end there)
    const int64_t total = dry_work;
    static double dry_ms, seg_ms, segs_n;
    static int64_t timed;
    dry_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_dry0).count();
    // (segments of ~RX_VK_PARALLEL_SEG draws and dispatches (default 160), taken in order by the threads and submitted
    // as each is done with those before it: the GPU starts on the first while the rest are recorded)
    static const int64_t per = getenv("RX_VK_PARALLEL_SEG") ? std::max(32, atoi(getenv("RX_VK_PARALLEL_SEG"))) : 160;
    const int nseg = (int)std::clamp<int64_t>(total / per, 1, kMaxBatch);
    std::vector<Segment> seg;
    seg.push_back({h, nullptr, &start, ub0, 0});
    for (int k = 1; k < nseg; k++) {
        const int64_t want = total * k / nseg;
        const ParPoint* at = nullptr;
        for (const ParPoint& q : par_points)
            if (q.work >= want) { at = &q; break; }
        if (!at || at->p == seg.back().p) continue;
        seg.back().end = at->p;
        seg.push_back({at->p, nullptr, &at->s, at->ub, at->cond});
    }
    seg.back().end = stop;
    if (seg.size() == 1) {   // (one segment: here, as the dry pass left the reports and conditions)
        record_segment(cmd, seg[0]);
        return stop;
    }
    if (!par_pool.started) {
        par_pool.started = true;
        for (int i = 0; i < threads; i++) std::thread(par_worker_main).detach();
        logf("vk: parallel replay: big command buffers recorded on %d threads in segments of ~%lld draws and dispatches",
             threads, (long long)per);
    }
    // what came before, to the GPU now (its uploads included); then each segment into a recording of its own (its
    // command buffer and uploads), recorded in order by the threads and submitted in order as they are done
    Recording& r = *rec_cur;
    end_pass(r.cmd);
    full_barrier(r.cmd);
    memory_sync_mirrors_before_segments();   // (in r's uploads, ahead of every segment)
    submit_recording(r, false);
    static uint64_t seq = 0;   // (the segments' order, across submissions: a recording's uploads precede it)
    const uint64_t base = ++seq;
    seq += seg.size();
    memory_rb_segments_begin(base, seg.size());   // (their order in the queue: memory_rb_refresh)
    struct Job {
        Recording* r;
        Segment g;
        uint64_t order;
        int index = 0;
        std::atomic<bool> done{false};
        bool uploads = false;
        int inits = 0;   // (texture_segment_end's)
    };
    std::vector<std::unique_ptr<Job>> jobs;
    for (size_t k = 0; k < seg.size(); k++) {
        auto j = std::make_unique<Job>();
        j->r = &begin_segment_recording(rec_cur_value);
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkResetCommandPool(dev.device, j->r->upload_pool, 0);
        vkBeginCommandBuffer(j->r->upload, &bi);
        j->g = seg[k];
        j->order = base + k;
        j->index = (int)k;
        jobs.push_back(std::move(j));
    }
    std::vector<VkCommandBuffer> inits;   // (each one's init command buffer, begun)
    {
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        for (auto& jp : jobs)
            if (jp->r->init && vkResetCommandPool(dev.device, jp->r->init_pool, 0) == VK_SUCCESS &&
                vkBeginCommandBuffer(jp->r->init, &bi) == VK_SUCCESS)
                inits.push_back(jp->r->init);
            else
                break;
    }
    if (inits.size() != jobs.size()) {
        // (one could not be begun: no worker has started, and the segments are not recorded. The whole command buffer is
        // recorded here as one segment, as a short one is, into what follows (the post-part as below): its new images are
        // made in queue order. The segments' recordings go unsubmitted; their slots come round again as unused ones)
        static int told;
        if (told++ < 8) logf("vk: parallel replay: a segment's init command buffer could not be begun: recorded on the replay thread");
        texture_set_segment(++seq);
        Recording& rest = begin_recording(rec_cur_value);
        full_barrier(rest.cmd);
        rec_cur = &rest;
        cmd = rest.cmd;
        rec_parts++;
        Segment whole = seg[0];
        whole.end = stop;
        record_segment(cmd, whole);
        return stop;
    }
    texture_segments_begin(inits.data(), inits.size());
    const auto t_seg0 = std::chrono::steady_clock::now();
    const int64_t wait0 = par_lock_wait_ns.load(), locks0 = par_locks.load();
    par_active = true;
    std::condition_variable seg_cv;
    std::mutex seg_mu;
    {
        std::lock_guard lk(par_pool.mu);
        for (auto& jp : jobs) {
            Job* j = jp.get();
            par_pool.jobs.push_back([j, &seg_cv, &seg_mu] {
                texture_set_segment(j->order);
                texture_segment_worker(j->index);
                texture_begin_recording(j->r->upload, j->r->value);
                draw_begin_recording(j->r->slot, j->r->cmd, j->r->value);
                VkCommandBuffer c = j->r->cmd;
                record_segment(c, j->g);
                j->uploads = texture_uploads_recorded();
                j->inits = texture_segment_end(j->index);   // (after its last new image)
                texture_segment_worker(-1);
                vkEndCommandBuffer(j->r->upload);
                vkEndCommandBuffer(j->r->cmd);
                {
                    std::lock_guard l(seg_mu);
                    j->done = true;
                }
                seg_cv.notify_all();
            });
            par_pool.pending++;
        }
    }
    par_pool.cv.notify_all();
    bool init_lost = false;
    for (auto& jp : jobs) {   // (in order, each as soon as it is done)
        Job* j = jp.get();
        {
            std::unique_lock l(seg_mu);
            seg_cv.wait(l, [j] { return j->done.load(); });
        }
        if (j->inits < 0) init_lost = true;   // (its init command buffer could not be ended: the replay stops, below)
        if (init_lost) continue;              // (nothing from it on is submitted)
        draw_flush_recording(j->r->slot);
        VkCommandBufferSubmitInfo cb[3]{{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO}, {VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO},
                                        {VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO}};
        uint32_t n = 0;
        if (j->inits > 0) cb[n++].commandBuffer = j->r->init;   // (new images made in it or after it: first in the queue)
        if (j->uploads) cb[n++].commandBuffer = j->r->upload;
        cb[n++].commandBuffer = j->r->cmd;
        VkSubmitInfo2 si{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
        si.commandBufferInfoCount = n;
        si.pCommandBufferInfos = cb;
        std::lock_guard lk(dev.present_mu);
        queue_flush_locked();
        const VkResult res = vkQueueSubmit2(dev.queue, 1, &si, j->r->fence);
        j->r->submitted = res == VK_SUCCESS;
        if (res != VK_SUCCESS) logf("vk: segment submit %llu failed (%d)", (unsigned long long)j->r->value, (int)res);
        if (res == VK_ERROR_DEVICE_LOST) device_lost();
    }
    {
        std::unique_lock lk(par_pool.mu);
        par_pool.done.wait(lk, [] { return par_pool.pending == 0; });
    }
    if (init_lost)   // (every worker done: none is left with a job)
        replay_fatal("a parallel replay segment's init command buffer could not be ended: new images lost their first layout "
                     "change, and the work after it may use them");
    par_active = false;
    texture_segments_begin(nullptr, 0);
    seg_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_seg0).count();
    segs_n += (double)seg.size();
    static double wait_ms, locks_n;
    wait_ms += (par_lock_wait_ns.load() - wait0) / 1e6, locks_n += (double)(par_locks.load() - locks0);
    if (++timed % 240 == 0) {
        logf("vk: parallel replay, a command buffer: dry pass %.2f ms, segments %.2f ms (%.1f of them), lock taken %.0f "
             "times, waited for %.2f ms (all threads)", dry_ms / 240, seg_ms / 240, segs_n / 240, locks_n / 240, wait_ms / 240);
        dry_ms = seg_ms = segs_n = wait_ms = locks_n = 0;
    }
    // what follows, after them all (the uniform updates are written to guest memory at the submission's end)
    texture_set_segment(++seq);
    Recording& nr = begin_recording(rec_cur_value);
    full_barrier(nr.cmd);
    rec_cur = &nr;
    cmd = nr.cmd;
    rec_parts += (int)seg.size() + 1;
    return stop;
}

// ---- the rx_* functions ----

// ---- a benchmark (RX_VK_BENCH=<present>[,<times>], default 50 times): the frame before that present replayed again and
// again, the game held (its next present waits) and the recordings kept (the replayed value the host frees them by
// is held at the frame's start); the replay's CPU time each time, logged. The same frame every time: what a change to
// the replay costs or saves, without the title scenes' differences. ----
std::vector<std::vector<uint64_t>> frame_submits;   // (this frame's submits' handles)
uint64_t frame_first_value = 0;
std::atomic<uint64_t> replayed_hold{0};             // (0: none)
bool stereo_held = false;                          // (VR stereo: replayed_hold is set for the frame's eyes)

void bench_replay(const std::vector<uint64_t>& handles) {
    const uint64_t value = pushed.load();   // (reached already: no slot waits on it)
    rec_value_now = value;
    ub_first_of_recording = ub_updates.size();
    Recording& first = begin_recording(value);
    memory_begin_submission();
    full_barrier(first.cmd);
    rec_cur = &first, rec_cur_value = value, rec_parts = 0, part_work = 0;
    VkCommandBuffer cmd = first.cmd;
    for (uint64_t h : handles)
        if (h) replay(cmd, *cur_state, reinterpret_cast<const uint8_t*>(h));
    Recording& r = *rec_cur;
    rec_cur = nullptr;
    end_pass(r.cmd);
    flush_ub_updates(r.cmd);
    memory_end_submission(r.cmd);
    full_barrier(r.cmd);
    submit_recording(r, false);
}

void bench_frame(int64_t presents) {
    static int64_t at = -2;
    static int times = 50;
    if (at == -2) {
        at = -1;
        if (const char* e = getenv("RX_VK_BENCH")) {
            at = atoll(e);
            if (const char* c = strchr(e, ',')) times = std::max(1, atoi(c + 1));
        }
    }
    if (at < 0 || presents + 1 != at || frame_submits.empty()) return;
    replayed_hold = std::max<uint64_t>(1, frame_first_value - 1);
    int64_t draws0 = 0, sk = 0, di = 0, pl = 0;
    draw_stats(&draws0, &sk, &di, &pl);
    std::vector<double> ms;
    int64_t prof0[PROF_COUNT];
    prof_full++;   // (every phase timed, not one draw in eight: the A/B differences are small)
    std::memcpy(prof0, prof_ns, sizeof prof0);
    const size_t ub_before = ub_updates.size();
    queue_flush();
    for (int i = 0; i < times; i++) {
        vkDeviceWaitIdle(dev.device);
        ub_updates.resize(ub_before);   // (the last time's uniform updates are written: as if pruned)
        const auto t0 = std::chrono::steady_clock::now();
        for (const auto& h : frame_submits) bench_replay(h);
        ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    vkDeviceWaitIdle(dev.device);
    prof_full--;
    int64_t draws1 = 0;
    draw_stats(&draws1, &sk, &di, &pl);
    replayed_hold = 0;
    std::vector<double> sorted = ms;
    std::sort(sorted.begin(), sorted.end());
    double sum = 0;
    for (double v : ms) sum += v;
    logf("vk: bench: the frame before present %lld (%zu submits, %lld draws) replayed %d times: median %.2f ms, min %.2f, "
         "mean %.2f, max %.2f", (long long)at, frame_submits.size(), (long long)((draws1 - draws0) / times), times,
         sorted[sorted.size() / 2], sorted.front(), sum / times, sorted.back());
    static const char* names[PROF_COUNT] = {"submit", "present", "draw", "descriptors", "texture lookups", "uploads",
                                            "pipelines", "programs", "commands", "dispatch", "memory sync",
                                            "queue submit", "slot wait"};
    char line[600];
    int n = snprintf(line, sizeof line, "vk: bench: a replay's time (ms):");
    for (int i = 2; i < PROF_COUNT; i++)
        n += snprintf(line + n, sizeof line - n, " %s %.2f%s", names[i], (prof_ns[i] - prof0[i]) / (double)times / 1e6,
                      i + 1 < PROF_COUNT ? "," : "");
    logf("%s", line);
}

/// A slow submit (over 15 ms): what took its time, logged (60 of them). The stalls in it are exact (PROF, PROF_EXACT); the
/// per-draw phases' cheap part is sampled, one draw in eight (PROF_HOT): an estimate, exact with RX_VK_PROF_ALL=1
struct SlowWatch {
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    int64_t prof0[PROF_COUNT];
    const char* what;
    explicit SlowWatch(const char* w) : what(w) { std::memcpy(prof0, prof_ns, sizeof prof0); }
    ~SlowWatch() {
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        static int logged;
        if (ms < 15.0 || logged >= 80 || presents_now < 600) return;   // (after the first load)
        logged++;
        static const char* names[PROF_COUNT] = {"submit", "present", "draw", "descriptors", "texture lookups",
                                                "uploads", "pipelines", "programs", "commands", "dispatch",
                                                "memory sync", "queue submit", "slot wait"};
        char line[600];
        int k = snprintf(line, sizeof line, "vk: slow %s: %.1f ms:", what, ms);
        for (int i = 2; i < PROF_COUNT; i++) {
            const double d = (prof_ns[i] - prof0[i]) / 1e6;
            if (d >= 1.0) k += snprintf(line + k, sizeof line - k, " %s %.1f", names[i], d);
        }
        logf("%s", line);
    }
};

/// The memory work before a submission's recording: the pools mapped, its capture taken, the other writes collected
/// (replay_mu held). In the order of the submits: one behind its queue's stopped one (submit_job) has it done as it comes.
void submit_memory_work(uint64_t value) {
    PROF(PROF_MEMSYNC);
    const auto m0 = std::chrono::steady_clock::now();
    memory_sync();
    // (the capture takes only the mirrored pools' pages, the collection the rest: skipping it whenever a capture was
    // taken, which with the replay thread is at every submit, left every other CPU write uncollected. Textures never
    // saw them: MK World's font cache kept its first glyphs (the menus lost letters), CPU-streamed terrain went stale)
    const auto m1 = std::chrono::steady_clock::now();
    memory_take_capture(value, completed());
    const auto m2 = std::chrono::steady_clock::now();
    memory_collect_written();
    const auto m3 = std::chrono::steady_clock::now();
    {   // (every 600 submits: what the memory work before a submit's recording cost)
        static double a, b, c;
        static int64_t k;
        a += std::chrono::duration<double, std::milli>(m1 - m0).count();
        b += std::chrono::duration<double, std::milli>(m2 - m1).count();
        c += std::chrono::duration<double, std::milli>(m3 - m2).count();
        if (++k % 600 == 0) {
            logf("vk: a submit's memory work (ms): maps %.2f, capture taken %.2f, writes collected %.2f", a / 600, b / 600, c / 600);
            a = b = c = 0;
        }
    }
}

// ---- the game's queues ordered by their events (RX_VK_EVENT_WAITS=0: every submit in the order it came, as before) ----
// MK World's two worker queues pass stages back and forth with events: a command buffer waits (nvnCommandBufferWaitEvent)
// for the other queue's to reach a stage, several times a frame, and the two are submitted one after the other.
// Replayed one after the other on the one Vulkan queue, the first went past waits for work the second had not done
// yet: what that work writes was read before it was written (55% of the waits). Now a submission is replayed as it
// comes as far as its waits allow, and what it recorded goes to the GPU; at a wait not met it stops, and its queue's
// later submits wait behind it (their memory work done in turn). Once another queue's work has signalled the event
// it goes on from there, as on the console, where it would have waited on the GPU. A present, a released texture or
// program, a wait from another thread for the replay, or RX_VK_HOLD_MS (default 3) with one stopped, and the stopped
// ones go on as their waits allow, the oldest past its wait whenever all of them wait. The timeline is signalled as
// far as no older submission is unfinished. (Holding a whole submission until the other queue's came, as this did
// first, had the registrations made meanwhile reach its draws before it: the menus' scenes in rectangles of fog.)
struct LaterSubmit {
    std::vector<uint64_t> handles;
    uint64_t value = 0;
};
struct QueueRun {   // (a queue stopped at a wait)
    bool waiting = false;
    std::vector<uint64_t> handles;   // (the submission it stopped in)
    int cb = 0;                      // (its command buffer, and the wait there)
    const uint8_t* p = nullptr;
    uint64_t value = 0;
    std::deque<LaterSubmit> later;   // (its queue's submits since)
    bool blocked() const { return waiting || !later.empty(); }
    uint64_t oldest() const { return waiting ? value : later.front().value; }
};
std::unordered_map<uint64_t, QueueRun> qruns;
std::atomic<int> blocked_n{0}, drain_waiters{0};    // (queues stopped, for the other threads; those in wait_queue_empty)
std::chrono::steady_clock::time_point blocked_since;
uint64_t done_max = 0, signaled = 0;                // (the newest value recorded whole; the timeline's last signal)
int64_t st_stops, st_resumed, st_forced_waits, st_forced[4], st_resume_captures;   // (made to go by: presents, releases, waits, time)
double st_stopped_max_ms;
bool event_waits_on() {
    static const bool v = !getenv("RX_VK_EVENT_WAITS") || atoi(getenv("RX_VK_EVENT_WAITS")) != 0;
    return v;
}
int hold_ms() {
    // (50: with the replay caught up, the worker queue's submission came up to 16 ms before the one that lets it on;
    // at 3 ms it went past its waits ~80 times a race. The game never waits for it before that submit)
    static const int v = getenv("RX_VK_HOLD_MS") ? std::max(0, atoi(getenv("RX_VK_HOLD_MS"))) : 50;
    return v;
}
const uint8_t* op_next(const uint8_t* p) {
    nvn_op_hdr h;
    std::memcpy(&h, p, sizeof h);
    return p + h.size;
}
nvn_p_wait_event wait_at(const uint8_t* p) {   // (p: a wait op)
    nvn_op_hdr h;
    std::memcpy(&h, p, sizeof h);
    return payload<nvn_p_wait_event>(p + 8, h.size - 8);
}
bool wait_met_at(const uint8_t* p) {
    const nvn_p_wait_event w = wait_at(p);
    ev_cpu_apply();
    return event_wait_met(event_value_in(ev_values, w.gpu), w.mode, w.value);
}
uint64_t oldest_unfinished() {
    uint64_t m = UINT64_MAX;
    for (const auto& [queue, q] : qruns)
        if (q.blocked()) m = std::min(m, q.oldest());
    return m;
}
// ---- (from PR 430, YunaraDebugs) Releases while queues are stopped: kept until the stopped work has gone on, instead of making it
// go at once (drain 1). Made to go early, a queue's work ran before the work it waits for, and about every 13th frame in Kokiri a
// pass read what was not there yet: the ground fog thinner for that frame, the light dimmed. The stopped work runs as before, at
// its wait, a present or hold_ms; the release follows it. RX_VK_RELEASE_DRAIN=1: as before ----
void do_texture_released(const nvn_tex* t);
void do_program_released(const nvn_prog* p);
std::mutex held_mu;   // (its own: sync_order_mu is held while whole submissions replay, and every release waited for them)
std::vector<nvn_tex> held_tex_releases;    // (held_mu)
std::vector<nvn_prog> held_prog_releases;  // (held_mu)
std::atomic<int64_t> st_held_releases{0};
bool hold_releases() {
    static const bool v = !getenv("RX_VK_RELEASE_DRAIN") || atoi(getenv("RX_VK_RELEASE_DRAIN")) == 0;
    return v;
}
void update_blocked() {
    int n = 0;
    for (const auto& [queue, q] : qruns) n += q.blocked();
    if (n && !blocked_n.load()) blocked_since = std::chrono::steady_clock::now();
    blocked_n = n;
    if (!n) {   // (the stopped work has gone on: its releases now)
        std::vector<nvn_tex> ts;
        std::vector<nvn_prog> ps;
        {
            std::lock_guard hl(held_mu);
            ts.swap(held_tex_releases);
            ps.swap(held_prog_releases);
        }
        for (const nvn_tex& t : ts) do_texture_released(&t);
        for (const nvn_prog& p : ps) do_program_released(&p);
    }
}
void log_order_stats() {
    logf("vk: queues ordered by their events: %lld submissions stopped at a wait, %lld went on once it was reached, %lld "
         "waits gone past; made to go by presents %lld, releases %lld, waits for the replay %lld, %d ms %lld (stopped "
         "%.2f ms at most); waits met %lld, not met %lld; captures at a CPU event's wait %lld", (long long)st_stops, (long long)st_resumed,
         (long long)st_forced_waits, (long long)st_forced[0], (long long)st_forced[1], (long long)st_forced[2], hold_ms(),
         (long long)st_forced[3], st_stopped_max_ms, (long long)st_waits_met, (long long)st_waits_unmet,
         (long long)st_resume_captures);
}

// ---- a burst: the pieces replayed one after the other with no memory work between them (a submit's job: its own
// piece and those of the stopped queues it lets on, then theirs again: MK World's two queues pass five or six stages
// back and forth a frame) go into one recording. Each piece had its own, and its memory sync, write-backs and submit:
// ~8 a frame. The burst's recording is tagged with the newest value among them (what is used in it is kept until
// that one is done; the timeline is signalled at its end as far as allowed, as for any). ----
bool burst_open = false;
uint64_t burst_value = 0;
void burst_begin(uint64_t value) {
    burst_open = event_waits_on() && !dump_on();
    burst_value = value;
    for (const auto& [queue, q] : qruns) {
        if (q.waiting) burst_value = std::max(burst_value, q.value);
        if (!q.later.empty()) burst_value = std::max(burst_value, q.later.back().value);
    }
}
void finish_recording(uint64_t value);
void burst_end() {
    if (!burst_open) return;
    burst_open = false;
    std::lock_guard lk(replay_mu);
    if (rec_cur) finish_recording(burst_value);
}

/// The recording being made: ended and submitted, the timeline signalled as far as no older submission is unfinished
void finish_recording(uint64_t value) {
    (void)value;
    Recording& r = *rec_cur;
    rec_cur = nullptr;
    end_pass(r.cmd);
    flush_ub_updates(r.cmd);
    memory_end_submission(r.cmd);
    full_barrier(r.cmd, "barrier (recording end)");
    const uint64_t oldest = oldest_unfinished();
    const uint64_t at = oldest == UINT64_MAX ? done_max : std::min(done_max, oldest - 1);
    const bool signal = at > signaled;
    if (signal) signaled = at;
    submit_recording(r, signal, at);
}

// ---- (hookshot) the synchronous replay on the core kept for the replay ----
// With RX_VK_ASYNC off (the default) a submission is replayed on the thread that submits it, or on whichever thread goes on
// with a queue stopped at an event wait, and the core the host keeps free of the game's threads for the replay thread
// (host->reserved_cpus: hos_reserved_cpus, the last core of the fastest kind) had nothing on it. Now a thread runs there
// while it replays and takes a submission's capture, and goes back to its own processors after: every thread that replays,
// none by name; with RX_VK_ASYNC=1 the replay thread is there already. Ryzen 9 3900X, Kokiri Forest: see the PR.
// RX_VK_REPLAY_PIN=0: off. (Only preferring the core, as the thread's ideal processor with every processor allowed, gained
// nothing there: the pin is what helps)
bool async_on();
// (Windows only: on Linux a thread inherits the affinity of the one that makes it, and the back end's threads made on
// first use inside a replay (the capture's copy helpers, the shader workers, the pool imports...) would all have stayed on
// the reserved core. Windows gives a new thread the process's)
uint64_t replay_pin_mask() {
#if defined(_WIN32)
    static const uint64_t m = [] {
        const char* e = getenv("RX_VK_REPLAY_PIN");
        const uint64_t r = !(e && atoi(e) == 0) && host->reserved_cpus ? host->reserved_cpus() : 0;
        if (r) logf("vk: replays run on logical processors %#llx while they replay (RX_VK_REPLAY_PIN=0: off)", (unsigned long long)r);
        return r;
    }();
    return m;
#else
    return 0;
#endif
}
thread_local int replay_pin_depth = 0;
thread_local DWORD_PTR replay_pin_prev = 0;
struct ReplayPin {
    bool counted = false;
    ReplayPin() {
        if (async_on()) return;
        counted = true;
        if (replay_pin_depth++) return;   // (the outermost one: a submit's replay inside its capture's)
        if (const uint64_t m = replay_pin_mask()) replay_pin_prev = SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)m);
    }
    ~ReplayPin() {
        if (!counted || --replay_pin_depth) return;
        if (replay_pin_prev) SetThreadAffinityMask(GetCurrentThread(), replay_pin_prev), replay_pin_prev = 0;
    }
    ReplayPin(const ReplayPin&) = delete;
    ReplayPin& operator=(const ReplayPin&) = delete;
};

/// A submission's command buffers from command buffer cb (op p; null: its start) on, into recordings of its own,
/// submitted: true when it finished; false when it stopped at a wait not met (may_stop), its place kept in qruns.
/// memory_work: its memory work first (else done as it came)
bool run_submit(const uint64_t* handles, int n, uint64_t value, uint64_t queue, bool memory_work, int cb = 0,
                const uint8_t* p = nullptr, bool may_stop = false) {
    std::lock_guard lk(replay_mu);
    ReplayPin pin;
    SlowWatch slow("submit");
    PROF(PROF_SUBMIT);
    if (memory_work) {
        const bool first_submit = frame_submits.empty();
        if (first_submit) {
            frame_first_value = value;
            vr::stereo_frame_start();   // (VR stereo: the camera is looked for in this frame; its recordings are kept for the eyes)
            if (vr::stereo_active()) stereo_held = true;
        }
        frame_submits.emplace_back(handles, handles + std::max(n, 0));
        submit_memory_work(value);
        if (first_submit) vr::stereo_prepass();
    }
    if (burst_open && rec_cur) {   // (the burst's recording goes on: what the piece before did, done first)
        end_pass(rec_cur->cmd);
        full_barrier(rec_cur->cmd, "barrier (a burst's next submit start)");
        use_queue_state(queue);
    } else {
        const uint64_t tag = burst_open ? std::max(burst_value, value) : value;
        use_queue_state(queue);
        ub_prune(completed());
        rec_value_now = tag;
        ub_first_of_recording = ub_updates.size();
        Recording& first = begin_recording(tag);
        memory_begin_submission();
        full_barrier(first.cmd, "barrier (recording start)");
        rec_cur = &first, rec_cur_value = tag, rec_parts = 0, part_work = 0;
    }
    VkCommandBuffer cmd = rec_cur->cmd;
    const uint8_t* stop = nullptr;
    for (; cb < n; cb++, p = nullptr) {
        if (!handles[cb]) continue;
        stop = replay_handle(cmd, p ? p : reinterpret_cast<const uint8_t*>(handles[cb]), may_stop);
        if (stop) break;
    }
    if (stop) {
        QueueRun& q = qruns[queue];
        q.waiting = true;
        q.handles.assign(handles, handles + n);
        q.cb = cb, q.p = stop, q.value = value;
        if (st_stops++ < 12) {
            const nvn_p_wait_event w = wait_at(stop);
            logf("vk: submission %llu (queue %#llx) stopped at command buffer %d's wait for event %#llx %s %u (has %u)",
                 (unsigned long long)value, (unsigned long long)queue, cb, (unsigned long long)w.gpu,
                 w.mode == 0 ? "==" : ">=", w.value, event_value_in(ev_values, w.gpu));
        }
        if (st_stops % 6000 == 0) log_order_stats();
    } else {
        done_max = std::max(done_max, value);
        rec_open_values.push_back(value);
    }
    if (!burst_open) finish_recording(value);   // (a burst's: at its end)
    if (stop) return false;
    if (pushed.load() < value) pushed = value;
    st_submits++;
    if (!burst_open && dump_pending()) wait_value(value), dump_write();
    return true;
}
void do_submit(const uint64_t* handles, int n) {   // (RX_VK_ASYNC=0: on the submitting thread, in its order)
    run_submit(handles, n, ++q_values, 0, true);
}

/// A stopped queue on (forced: past the wait it stopped at), then its later submits, as far as their waits allow
void resume_queue(uint64_t queue, bool forced) {
    QueueRun& q = qruns[queue];
    for (;;) {
        if (q.waiting) {
            // (hookshot) a wait for an event the CPU signals, met: the game wrote what this work reads after it submitted it,
            // up to the signal (OoT's presentation queue waits for its frame's data this way, 1 frame in 12 here). The
            // console's GPU reads memory after its wait; the capture taken at the submit had the frame before's: the fog and
            // exposure of the frame before, for a frame. A capture now, for what follows the wait. RX_VK_RESUME_CAPTURE=0: none
            static const bool resume_capture = !getenv("RX_VK_RESUME_CAPTURE") || atoi(getenv("RX_VK_RESUME_CAPTURE")) != 0;
            bool cpu_event = false;
            if (resume_capture && !forced) {
                std::lock_guard ek(ev_cpu_mu);
                cpu_event = cpu_events.count(wait_at(q.p).gpu) != 0;
            }
            if (cpu_event) {
                std::lock_guard lk(replay_mu);
                if (rec_cur) finish_recording(burst_open ? burst_value : q.value);   // (its mirrors synced again after)
                memory_capture_now(q.value, completed());
                st_resume_captures++;
            }
            q.waiting = false;
            std::vector<uint64_t> handles = std::move(q.handles);
            q.handles.clear();
            const uint8_t* from = forced ? op_next(q.p) : q.p;
            (forced ? st_forced_waits : st_resumed)++;
            if (!run_submit(handles.data(), (int)handles.size(), q.value, queue, false, q.cb, from, true)) return;
            continue;
        }
        if (q.later.empty()) return;
        LaterSubmit s = std::move(q.later.front());
        q.later.pop_front();
        if (!run_submit(s.handles.data(), (int)s.handles.size(), s.value, queue, false, 0, nullptr, true)) return;
    }
}
/// The stopped queues whose events are reached go on, until none can
void schedule() {
    for (bool moved = true; moved;) {
        moved = false;
        for (auto& [queue, q] : qruns)
            if (q.waiting && wait_met_at(q.p)) {
                resume_queue(queue, false);
                moved = true;
            }
    }
}
/// Every stopped queue on now (why: an index of st_forced): in turn as their waits allow; when all of them wait, the
/// oldest past its wait
void drain(int why) {
    if (oldest_unfinished() == UINT64_MAX) return;
    st_forced[why]++;
    st_stopped_max_ms = std::max(st_stopped_max_ms,
                                 std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - blocked_since).count());
    burst_begin(0);
    for (;;) {
        schedule();
        uint64_t best = UINT64_MAX, bq = 0;
        for (const auto& [queue, q] : qruns)
            if (q.blocked() && q.oldest() < best) best = q.oldest(), bq = queue;
        if (best == UINT64_MAX) break;
        resume_queue(bq, qruns[bq].waiting);
    }
    burst_end();
    update_blocked();
}

// ---- (hookshot) the same on the submitting thread, the default. do_submit replayed each queue's submission as it came and went
// past every wait: 2 of 3 were not met. A lit pass waiting for another queue's work that had not been submitted yet read what
// that work makes before it was made, and for a frame lit surfaces came out black (a race with the game's threads: 3-9% of
// frames with frame generation off, less the less they overlapped). The stopped work goes on from whichever thread's submit or
// signal lets it, or a present, a wait for it, or hold_ms (sync_watch: nothing else may come while the game polls).
// RX_VK_SYNC_WAITS=0: as before ----
bool async_on();
/// RX_VK_PER_QUEUE_FENCES=0: a fence waits for every queue's work up to it, stopped work forced on, as before
bool per_queue_fences_off() {
    static const bool v = getenv("RX_VK_PER_QUEUE_FENCES") && atoi(getenv("RX_VK_PER_QUEUE_FENCES")) == 0;
    return v;
}
std::mutex sync_order_mu;   // (before replay_mu, always; all of the order's state: qruns, burst_*)
bool sync_waits_on() {
    static const bool v = !getenv("RX_VK_SYNC_WAITS") || atoi(getenv("RX_VK_SYNC_WAITS")) != 0;
    return v && event_waits_on() && !async_on() && !dump_on();
}
void sync_drain(int why) {
    if (!blocked_n.load()) return;
    std::lock_guard lk(sync_order_mu);
    drain(why);
    update_blocked();
}
/// (hookshot) The stopped queues whose waits are met go on; none is forced past its wait
void sync_settle() {
    if (!blocked_n.load()) return;
    std::lock_guard lk(sync_order_mu);
    burst_begin(0);
    schedule();
    burst_end();
    update_blocked();
}
void sync_watch() {
    SetThreadDescription(GetCurrentThread(), L"mkw_vk order");
    for (;;) {
        Sleep(1);
        if (!blocked_n.load()) continue;
        std::unique_lock lk(sync_order_mu, std::try_to_lock);
        if (!lk) continue;
        burst_begin(0);   // (a signal from the CPU the thread that made it could not act on)
        schedule();
        burst_end();
        update_blocked();
        const double held = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - blocked_since).count();
        if (blocked_n.load() && held > hold_ms()) {
            drain(3);
            update_blocked();
        }
    }
}

/// A submit on the replay thread: replayed as far as its waits allow, or (its queue stopped) after its queue's
void submit_job(std::vector<uint64_t>& handles, uint64_t value, uint64_t queue) {
    const bool may_stop = event_waits_on() && !dump_on();
    if (!may_stop) drain(2);
    auto it = qruns.find(queue);
    if (it != qruns.end() && it->second.blocked()) {   // (its memory work now, in the order of the submits)
        std::lock_guard lk(replay_mu);
        const bool first_submit = frame_submits.empty();
        if (first_submit) {
            frame_first_value = value;
            vr::stereo_frame_start();
            if (vr::stereo_active()) stereo_held = true;
        }
        frame_submits.emplace_back(handles);
        submit_memory_work(value);
        if (first_submit) vr::stereo_prepass();
        it->second.later.push_back({std::move(handles), value});
        return;
    }
    if (may_stop) burst_begin(value);   // (its piece and what it lets on: one recording)
    run_submit(handles.data(), (int)handles.size(), value, queue, true, 0, nullptr, may_stop);
    schedule();   // (what it signalled may let the stopped ones on)
    burst_end();
}
uint64_t get_replayed() {
    queue_flush_poll();
    const uint64_t hold = replayed_hold.load();
    return hold ? std::min(hold, completed()) : completed();
}
/// (hookshot) Waits for the recording submission value f (or the recording of value f) was finished in, on the recordings'
/// timeline: true when that recording is known (the result in *ok), false when it is not (f not finished yet)
bool wait_recorded(uint64_t f, uint64_t timeout_ns, int* ok) {
    if (!rec_timeline || per_queue_fences_off()) return false;
    uint64_t seq = 0;
    {
        std::lock_guard mk(rec_map_mu);
        auto it = value_rec.find(f);
        if (it == value_rec.end()) return false;
        seq = it->second;
    }
    // (a batched submit holds its recordings' signal until the batch goes out: the readback wait in invalidate came here
    // without a flush, and the job that waited held up the main thread, so nothing flushed it: the load hang on RADV)
    queue_flush_for(FR_WAIT);
    VkSemaphoreWaitInfo wi{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
    wi.semaphoreCount = 1;
    wi.pSemaphores = &rec_timeline;
    wi.pValues = &seq;
    *ok = vkWaitSemaphores(dev.device, &wi, timeout_ns) == VK_SUCCESS;
    return true;
}
uint64_t fence();
/// (hookshot) A fence of one queue (nvnQueueFenceSync): the value of that queue's last submission
uint64_t fence_q(uint64_t queue) {
    if (async_on() || !sync_waits_on() || per_queue_fences_off() || !queue) return fence();
    std::lock_guard lk(sync_order_mu);
    auto it = q_last.find(queue);
    return it != q_last.end() ? it->second : fence();
}
int wait(uint64_t f, uint64_t timeout_ns) {
    if (completed() >= f) return 1;
    queue_flush_for(FR_WAIT);
    // (hookshot) a fence of one queue whose work up to it is recorded whole (none of it stopped): that work, not the other
    // queues' stopped work the timeline above waits for (which a wait for it forced past its waits after RX_VK_WAIT_HOLD_US)
    if (!async_on() && sync_waits_on() && !per_queue_fences_off()) {
        bool mine = false;
        {
            std::lock_guard lk(sync_order_mu);
            auto it = value_queue.find(f);
            if (it != value_queue.end()) {
                auto q = qruns.find(it->second);
                mine = q == qruns.end() || !q->second.blocked() || q->second.oldest() > f;
            }
        }
        int ok = 0;
        if (mine && wait_recorded(f, timeout_ns, &ok)) return ok;
    }
    // (a wait that needs stopped work: another queue's submit usually lets that on in a moment. Forced at once, as the main thread
    // waits ~15 times a second, 69000 waits went past in five minutes, and each one that mattered was a black frame.
    // RX_VK_WAIT_HOLD_US: how long the waiter gives it. A wait for older work than the stopped ones needs nothing of them)
    if (timeout_ns && !async_on() && blocked_n.load()) {
        static const int64_t hold_ns = (getenv("RX_VK_WAIT_HOLD_US") ? atoll(getenv("RX_VK_WAIT_HOLD_US")) : 3000) * 1000;
        const auto t0 = std::chrono::steady_clock::now();
        for (;;) {
            queue_flush_for(FR_WAIT);
            if (completed() >= f) return 1;
            bool needs;
            {
                std::lock_guard lk(sync_order_mu);
                needs = oldest_unfinished() <= f;
            }
            if (!needs) break;
            const int64_t waited = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
            if ((uint64_t)waited >= timeout_ns) break;   // (sync_watch's, past hold_ms)
            if (waited >= hold_ns) {
                sync_drain(2);
                break;
            }
            std::this_thread::yield();
        }
    }
    VkSemaphoreWaitInfo wi{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
    wi.semaphoreCount = 1;
    wi.pSemaphores = &timeline;
    wi.pValues = &f;
    queue_flush_for(FR_WAIT);
    const VkResult res = vkWaitSemaphores(dev.device, &wi, timeout_ns);
    static int logged;
    if (res != VK_SUCCESS && timeout_ns > 1000000000ull && logged++ < 8)
        logf("vk: wait for fence %llu (done %llu, pushed %llu): %d", (unsigned long long)f,
             (unsigned long long)completed(), (unsigned long long)pushed.load(), (int)res);
    if (res == VK_ERROR_DEVICE_LOST) device_lost();
    return res == VK_SUCCESS;
}
void wait_queue_empty();
bool replay_behind();
void finish() {
    wait_queue_empty();
    wait_value(pushed.load());
}
void wait_writebacks_reading(const void*, uint64_t) {}   // (nothing to write back: the GPU writes guest memory)
/// (host: the CPU is about to read [gpu, gpu + size), nvnMemoryPoolInvalidateMappedRange / nvnBufferInvalidateMappedRange)
/// RX_VK_LOG_INVALIDATE=1: the ranges, 400 lines, and the textures the GPU wrote there (their texels are not in memory)
// ---- live textures' storage, by 4 KB page (as the host made and finalized them, ahead of the replay) ----
// For invalidate: the CPU reads a range back; texels the GPU wrote into a texture there are not in memory until a
// write-back, and a texture the replay has not listed yet (it runs behind) needs it to catch up first. Ranges no live
// texture's storage touches hold buffers only, which the submissions write back themselves: no catch-up for them.
// (Counts per page: storage shared by several textures; two levels, chunks of 1M pages made when first touched.)
constexpr uint64_t kTexPageChunk = 1u << 20;
std::unique_ptr<std::atomic<uint16_t>[]> tex_pages[(1ull << 39 >> 12) / kTexPageChunk];
std::mutex tex_pages_mu;
void tex_pages_add(const nvn_tex* t, int d) {
    if (!t || !t->gpu || t->gpu >= (1ull << 39) || !t->L.size) return;   // (the guest window: virtual pools are not read)
    const uint64_t p0 = t->gpu >> 12, p1 = std::min((t->gpu + t->L.size - 1) >> 12, uint64_t((1ull << 39 >> 12) - 1));
    for (uint64_t p = p0; p <= p1; p++) {
        auto& c = tex_pages[p / kTexPageChunk];
        if (!c) {
            std::lock_guard lk(tex_pages_mu);
            if (!c) {
                auto n = std::make_unique<std::atomic<uint16_t>[]>(kTexPageChunk);
                for (uint64_t i = 0; i < kTexPageChunk; i++) n[i].store(0, std::memory_order_relaxed);
                c = std::move(n);
            }
        }
        std::atomic<uint16_t>& v = c[p % kTexPageChunk];
        if (d > 0) v.fetch_add(1, std::memory_order_relaxed);
        else if (v.load(std::memory_order_relaxed)) v.fetch_sub(1, std::memory_order_relaxed);
    }
}
bool tex_pages_any(uint64_t gpu, uint64_t size) {
    if (gpu >= (1ull << 39)) return true;   // (not tracked: as before)
    const uint64_t p0 = gpu >> 12, p1 = std::min((gpu + size - 1) >> 12, uint64_t((1ull << 39 >> 12) - 1));
    for (uint64_t p = p0; p <= p1; p++) {
        const auto& c = tex_pages[p / kTexPageChunk];
        if (!c) { p = (p / kTexPageChunk + 1) * kTexPageChunk - 1; continue; }
        if (c[p % kTexPageChunk].load(std::memory_order_relaxed)) return true;
    }
    return false;
}
void texture_created_of(const nvn_tex* t) { tex_pages_add(t, +1); }
uint64_t fence();
void invalidate(uint64_t gpu, uint64_t size) {
    static const bool log_inv = getenv("RX_VK_LOG_INVALIDATE") != nullptr;
    static std::atomic<int> logged{0};
    if (log_inv && logged++ < 400) {
        std::string hits;
        {
            std::lock_guard lk(replay_mu);
            texture_describe_written(gpu, size, hits);
        }
        logf("vk: invalidate %#llx +%#llx:%s", (unsigned long long)gpu, (unsigned long long)size,
             hits.empty() ? " no texture the GPU wrote" : hits.c_str());
    }
    {   // (EvilBoyce, diagnostics) RX_VK_LOG_READBACK=<present>: from that present on, 300 invalidates with the fence handed
        // out, the GPU's completed value, whether the replay is behind, and the first words the CPU is about to read
        static const int64_t from = getenv("RX_VK_LOG_READBACK") ? atoll(getenv("RX_VK_LOG_READBACK")) : -1;
        static std::atomic<int> n{0};
        if (from >= 0 && presents_now >= from && n++ < 300) {
            uint32_t w[8]{};
            host->gpu_read(gpu, w, sizeof w);
            logf("vk: readback %lld %#llx +%#llx: fence %llu done %llu behind %d | %08x %08x %08x %08x %08x %08x %08x %08x",
                 (long long)presents_now, (unsigned long long)gpu, (unsigned long long)size, (unsigned long long)fence(),
                 (unsigned long long)completed(), (int)replay_behind(), w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]);
        }
    }
    // (EvilBoyce) GPU results in buffers: the CPU reads them once the submission that writes them back is done. The game
    // reads them when its own pacing says the GPU has finished that frame, which on the console's coherent memory means the
    // values are there; here they reach guest memory only with the write-back at the end of the submission that the replay
    // may not have recorded or the GPU not finished yet. OoT reads its exposure meter (a sum the GPU adds to each frame and
    // the CPU reads and sets to 0) this way: it read 0 one frame, then the old sum came back over the CPU's 0 and the next
    // frame added to it (2x): the exposure stepped 0.2 EV up and down several times a second (the white flashes).
    // RX_VK_NO_READBACK_WAIT=1: as before
    if (size) {
        static const bool off = getenv("RX_VK_NO_READBACK_WAIT") != nullptr;
        bool ever = false;
        if (!off && (memory_writeback_value(host->gpu_guest(gpu), size, &ever), ever)) {
            memory_note_readback(host->gpu_guest(gpu), size);   // (PR 430: the CPU may write it back: memory_rb_refresh)
            static std::atomic<int64_t> calls{0}, waited{0}, ns{0};
            const auto w0 = std::chrono::steady_clock::now();
            bool w = false;
            const uint64_t f0 = fence(), c0 = completed();
            const bool behind0 = replay_behind();
            static const bool no_catchup = getenv("RX_VK_READBACK_NO_CATCHUP") != nullptr;   // (A/B: recorded work only)
            // (hookshot) not on the submitting thread's replay (the default): there what is handed over is recorded at once, and
            // the work not recorded yet is stopped at an event wait. Catching up forced it past that wait, ahead of the work it
            // waits for (OoT's presentation work before the scene's: the fog of the frame before, for a frame, at 1 in 20 of the
            // exposure readbacks). The write-back is waited for on the recordings' timeline instead, which a stopped queue
            // does not hold back
            const bool unforced = !async_on() && sync_waits_on() && !per_queue_fences_off();
            if (behind0 && !no_catchup && !unforced) wait_queue_empty(), w = true;   // (work handed over may write it back too)
            const auto w1 = std::chrono::steady_clock::now();
            const uint64_t v = memory_writeback_value(host->gpu_guest(gpu), size, nullptr);
            if (v > completed()) {
                int ok = 0;
                if (!(unforced && wait_recorded(v, UINT64_MAX, &ok))) {
                    if (unforced && replay_behind()) wait_queue_empty();   // (not recorded whole: as before)
                    wait_value(v);
                }
                w = true;
            }
            if (w) {
                waited++;
                const int64_t d = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - w0).count();
                ns += d;
                static const bool log_each = getenv("RX_VK_LOG_READBACK_WAITS") != nullptr;
                static std::atomic<int> logged{0};
                if (log_each && presents_now > 4000 && logged++ < 300)
                    logf("vk: readback wait %lld %#llx +%#llx: %.2f ms (catch-up %.2f) value %llu, fence %llu, done %llu, behind %d",
                         (long long)presents_now, (unsigned long long)gpu, (unsigned long long)size, d / 1e6,
                         std::chrono::duration_cast<std::chrono::nanoseconds>(w1 - w0).count() / 1e6, (unsigned long long)v,
                         (unsigned long long)f0, (unsigned long long)c0, (int)behind0);
            }
            if (++calls % 3000 == 0) {
                const int64_t n = waited.exchange(0);
                logf("vk: readbacks of GPU results: %lld of 3000 waited for their write-back, %.2f ms each", (long long)n,
                     n ? ns.exchange(0) / (double)n / 1e6 : 0.0);
            }
        }
    }
    // (textures the GPU wrote there: their texels into memory before the CPU reads it, as vmap_changing does. The game
    // waited for the work that wrote them; the replay is caught up first)
    {   // (RX_VK_READBACK_OFF_FILE=<path> (A/B): while that file exists, not)
        static const char* off_file = getenv("RX_VK_READBACK_OFF_FILE");
        if (off_file && GetFileAttributesA(off_file) != INVALID_FILE_ATTRIBUTES) return;
    }
    // (the replay caught up first only for ranges a texture the GPU wrote may be in: MKW invalidates a few buffer
    // ranges every frame (GPU results it reads back, written back with their submission; one of them a ring of
    // slots at a new address each time), and the catch-up for each held its main thread until the replay had
    // recorded everything queued: ~10% of the main thread's time. A 1 MB region caught up for kProbe times without a
    // texture found holds buffers only from then on; one a texture was ever found in keeps the catch-up.
    // RX_VK_INVALIDATE_WAIT=1: every range, as before)
    // (decided by the live textures' storage, not by what the replay has listed: a region guessed to hold buffers
    // only, after probes that found no texture, left MKW's Grand Prix waiting at its map for a texture's texels)
    static const bool always = getenv("RX_VK_INVALIDATE_WAIT") != nullptr;
    if (!size) return;
    // (hookshot) and on the submitting thread's replay, the catch-up lets on what may go without forcing anything past its wait
    // (once the exposure readback above no longer did, this one forced OoT's presentation work on, every frame or two)
    const bool unforced = !async_on() && sync_waits_on() && !per_queue_fences_off();
    if (!texture_gpu_written_in(gpu, size)) {
        if (!replay_behind()) return;
        if (!always && !tex_pages_any(gpu, size)) return;   // (no texture's storage there: buffers only)
        unforced ? sync_settle() : wait_queue_empty();
        if (!texture_gpu_written_in(gpu, size)) return;
    }
    {   // (what the catch-ups for a texture cost the game's thread, every 300 calls)
        static std::atomic<int64_t> calls{0}, ns{0};
        const auto w0 = std::chrono::steady_clock::now();
        unforced ? sync_settle() : wait_queue_empty();
        ns += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - w0).count();
        if (++calls % 300 == 0)
            logf("vk: invalidate %#llx +%#llx: textures there; the catch-ups waited %.2f ms each (300 calls)",
                 (unsigned long long)gpu, (unsigned long long)size, ns.exchange(0) / 300.0 / 1e6);
    }
    std::lock_guard lk(replay_mu);
    texture_flush_to_guest(gpu, size);
}
void do_texture_released(const nvn_tex* t) {
    std::lock_guard lk(replay_mu);
    if (t) {
        mkwvk::texture_released(t->serial);   // (vk_bank.cpp: the registrations of it)
        mkwvk::texture_released(t);           // (vk_texture.cpp: its image may take CPU writes again)
    }
}

// ---- presents ----
struct Frame {
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkFence done = VK_NULL_HANDLE;
    VkSemaphore acquired = VK_NULL_HANDLE;
};
Frame frames[3];
uint32_t frame_index = 0;
int64_t presents = 0;

// offscreen (no window): the picture, and the readback of a screenshot
VkImage off_image = VK_NULL_HANDLE;
VmaAllocation off_alloc = nullptr;
// (two readbacks, by the present's parity: a shot is read at a later present, after that one's own copy was submitted,
// and with one buffer the CPU read it while the GPU wrote the next frame over it: torn shots, and flashes counted)
VkBuffer shot_bufs[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
VmaAllocation shot_allocs[2] = {nullptr, nullptr};
void* shot_maps[2] = {nullptr, nullptr};
// (RX_VK_SHOT_W / RX_VK_SHOT_H: another size, e.g. the game's output size for its own pixels)
uint32_t shot_dimension(const char* text, uint32_t fallback) {
    if (!text) return fallback;
    if (!*text) return 0;
    uint32_t value = 0;
    for (; *text; text++) {
        if (*text < '0' || *text > '9') return 0;
        const uint32_t digit = (uint32_t)(*text - '0');
        if (value > (UINT32_MAX - digit) / 10) return 0;
        value = value * 10 + digit;
    }
    return value;
}
const uint32_t kShotW = shot_dimension(getenv("RX_VK_SHOT_W"), 1280),
               kShotH = shot_dimension(getenv("RX_VK_SHOT_H"), 720);
VkImage shot_image = VK_NULL_HANDLE;
VkImageView shot_view = VK_NULL_HANDLE;   // (hookshot: RX_VK_SHOT_MENU's menu draws through it)
VkBuffer made_buf = VK_NULL_HANDLE;   // (RX_VK_SHOT_MADE: a made frame's readback)
VmaAllocation made_buf_alloc = nullptr;
void* made_map = nullptr;
VkImage made_image = VK_NULL_HANDLE;
VmaAllocation made_image_alloc = nullptr;
VmaAllocation shot_image_alloc = nullptr;

bool shot_dimensions_ready() {
    static int state = 0;
    if (state) return state > 0;
    state = -1;
    const uint32_t limit = dev.props.limits.maxImageDimension2D;
    const uint64_t pixels = (uint64_t)kShotW * kShotH, row = (uint64_t)kShotW * 3 + 1;
    bool valid = false;
    if (kShotW && kShotH && kShotW <= limit && kShotH <= limit && pixels <= SIZE_MAX / 4) {
        const uint64_t raw = row * kShotH;
        // write_png emits one IDAT: zlib header, stored blocks (5 bytes per65535), and Adler32.
        // PNG chunks are limited to2^31-1 bytes. This is an encoder limit, not a memory budget.
        constexpr uint64_t max_chunk = 0x7fffffffu;
        if (raw <= max_chunk) {
            const uint64_t encoded = 2 + raw + 5 * ((raw + 65534) / 65535) + 4;
            valid = encoded <= max_chunk && encoded <= SIZE_MAX - 4;
        }
    }
    if (!valid) {
        logf("vk: screenshots disabled: RX_VK_SHOT_W/H must be positive decimal dimensions within the device limit "
             "(%u) and the PNG/readback size limits (got %ux%u)", limit, kShotW, kShotH);
        return false;
    }
    state = 1;
    return true;
}

bool made_shot_init() {
    if (made_image) return true;
    VkImage next_image = VK_NULL_HANDLE;
    VmaAllocation next_alloc = nullptr, next_buf_alloc = nullptr;
    VkBuffer next_buf = VK_NULL_HANDLE;
    void* next_map = nullptr;
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = VK_FORMAT_R8G8B8A8_UNORM;
    ci.extent = {kShotW, kShotH, 1};
    ci.mipLevels = ci.arrayLayers = 1;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    if (!create_image(ci, next_image, next_alloc)) return false;
    if (!create_buffer((uint64_t)kShotW * kShotH * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true,
                       next_buf, next_buf_alloc, &next_map)) {
        destroy_image(next_image, next_alloc);
        return false;
    }
    made_image = next_image, made_image_alloc = next_alloc;
    made_buf = next_buf, made_buf_alloc = next_buf_alloc, made_map = next_map;
    return true;
}

bool offscreen_init() {
    if (!shot_dimensions_ready()) return false;
    if (off_image) return true;
    VkImage next_off = VK_NULL_HANDLE, next_shot = VK_NULL_HANDLE;
    VkImageView next_view = VK_NULL_HANDLE;
    VmaAllocation next_off_alloc = nullptr, next_shot_alloc = nullptr, next_allocs[2]{};
    VkBuffer next_bufs[2]{};
    void* next_maps[2]{};
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = VK_FORMAT_R8G8B8A8_UNORM;
    ci.extent = {dev.swap_extent.width, dev.swap_extent.height, 1};
    ci.mipLevels = ci.arrayLayers = 1;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    bool ok = create_image(ci, next_off, next_off_alloc);
    ci.extent = {kShotW, kShotH, 1};
    ci.usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;   // (hookshot: RX_VK_SHOT_MENU draws the menu over it, vk_menu.cpp)
    ok = ok && create_image(ci, next_shot, next_shot_alloc);
    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = next_shot;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = ci.format;
    vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    ok = ok && vkCreateImageView(dev.device, &vi, nullptr, &next_view) == VK_SUCCESS;
    for (int i = 0; i < 2; i++)
        ok = ok && create_buffer((uint64_t)kShotW * kShotH * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true,
                                 next_bufs[i], next_allocs[i], &next_maps[i]);
    if (!ok) {
        for (int i = 0; i < 2; i++)
            if (next_bufs[i]) destroy_buffer(next_bufs[i], next_allocs[i]);
        if (next_view) vkDestroyImageView(dev.device, next_view, nullptr);
        if (next_shot) destroy_image(next_shot, next_shot_alloc);
        if (next_off) destroy_image(next_off, next_off_alloc);
        return false;
    }
    // Publish the ready marker only after both images, the menu view and both mapped readbacks exist.
    off_image = next_off, off_alloc = next_off_alloc;
    shot_image = next_shot, shot_image_alloc = next_shot_alloc;
    shot_view = next_view;
    for (int i = 0; i < 2; i++)
        shot_bufs[i] = next_bufs[i], shot_allocs[i] = next_allocs[i], shot_maps[i] = next_maps[i];
    return true;
}

void image_barrier(VkCommandBuffer cmd, VkImage image, VkImageLayout from, VkImageLayout to) {
    VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    b.srcStageMask = b.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    b.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
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

// A completed capture copy must also be made available in the host memory domain.
void capture_host_read_barrier(VkCommandBuffer cmd, VkBuffer buffer, VkDeviceSize bytes) {
    VkBufferMemoryBarrier2 b{};
    b.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
    b.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    b.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    b.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
    b.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.buffer = buffer;
    b.size = bytes;
    VkDependencyInfo d{};
    d.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    d.bufferMemoryBarrierCount = 1;
    d.pBufferMemoryBarriers = &b;
    vkCmdPipelineBarrier2(cmd, &d);
}

/// The crop of src (x, y, w, h; w 0: all of it) scaled into a rectangle of dst.
/// An empty source clears the whole destination image; current callers give that fallback a full-image rectangle.
void blit_crop_to_rect(VkCommandBuffer cmd, VkImage src, uint32_t sw, uint32_t sh, const int crop[4], VkImage dst,
                       VkImageLayout dst_layout, PresentRect rect) {
    const PresentSourceRect source = present_source_rect(sw, sh, crop);
    if (!rect.w || !rect.h) return;
    if (!source.w || !source.h) {
        const VkClearColorValue black{{0.0f, 0.0f, 0.0f, 1.0f}};
        const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(cmd, dst, dst_layout, &black, 1, &range);
        return;
    }
    VkImageBlit b{};
    b.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    b.srcOffsets[0] = {(int)source.x, (int)source.y, 0};
    b.srcOffsets[1] = {(int)(source.x + source.w), (int)(source.y + source.h), 1};
    b.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    b.dstOffsets[0] = {(int)rect.x, (int)rect.y, 0};
    b.dstOffsets[1] = {(int)(rect.x + rect.w), (int)(rect.y + rect.h), 1};
    vkCmdBlitImage(cmd, src, VK_IMAGE_LAYOUT_GENERAL, dst, dst_layout, 1, &b, VK_FILTER_LINEAR);
}

/// The crop of src scaled to the whole destination (captures keep their existing dimensions).
void blit_crop(VkCommandBuffer cmd, VkImage src, uint32_t sw, uint32_t sh, const int crop[4], VkImage dst,
               VkImageLayout dst_layout, uint32_t dw, uint32_t dh) {
    blit_crop_to_rect(cmd, src, sw, sh, crop, dst, dst_layout, {0, 0, dw, dh});
}

// ---- PNG (stored deflate blocks: no compression library needed) ----
uint32_t crc32(const uint8_t* p, size_t n, uint32_t c = 0xFFFFFFFFu) {
    static const std::array<uint32_t, 256> table = [] {
        std::array<uint32_t, 256> out{};
        for (uint32_t i = 0; i < 256; i++) {
            uint32_t k = i;
            for (int j = 0; j < 8; j++) k = k & 1 ? 0xEDB88320u ^ (k >> 1) : k >> 1;
            out[i] = k;
        }
        return out;
    }();
    for (size_t i = 0; i < n; i++) c = table[(c ^ p[i]) & 0xFF] ^ (c >> 8);
    return c;
}
} // namespace

/// A submission or a wait came back VK_ERROR_DEVICE_LOST. Nothing sent from here on runs, and the replay kept going: no
/// submission completed again, so no capture or staging arena was let go, and system memory filled until the window was
/// closed (issue 1275: captures kept 25 -> 6527, staging 256 -> 1408 MB). Now the report once (a thread that sees the loss
/// too waits for it, instead of ending the process before it is written), then the process ends as replay_fatal does
[[noreturn]] void device_lost() {
    static std::once_flag once;
    std::call_once(once, checkpoints_report);
    replay_fatal("the GPU device was lost (VK_ERROR_DEVICE_LOST): the driver stopped or reset the GPU, and nothing sent "
                 "to it runs any more (on Linux, dmesg or journalctl -k around this time names the reset)");
}

bool write_png(const char* path, const uint8_t* rgba, uint32_t w, uint32_t h) {
    auto failed = [path](const char* phase, unsigned error = 0) {
        static std::atomic<unsigned> reported{0};
        if (reported.fetch_add(1, std::memory_order_relaxed) < 8)
            logf("vk: PNG %s failed (error %u): %s", phase, error, path ? path : "(null)");
        return false;
    };
    if (!path || !*path || !rgba || !w || !h || w > 0x7fffffffu || h > 0x7fffffffu)
        return failed("dimensions/path");
    const uint64_t pixels = (uint64_t)w * h;
    if (pixels > SIZE_MAX / 4) return failed("size");
    const uint64_t raw_size = ((uint64_t)w * 3 + 1) * h;
    constexpr uint64_t max_chunk = 0x7fffffffu;
    if (raw_size > max_chunk || 6 + raw_size + 5 * ((raw_size + 65534) / 65535) > max_chunk)
        return failed("size");
    try {
        std::vector<uint8_t> raw;
        raw.reserve((size_t)(w * 3 + 1) * h);
        for (uint32_t y = 0; y < h; y++) {
            raw.push_back(0);
            for (uint32_t x = 0; x < w; x++) {
                const uint8_t* q = rgba + ((size_t)y * w + x) * 4;
                raw.insert(raw.end(), q, q + 3);
            }
        }
        std::vector<uint8_t> z = {0x78, 0x01};
        uint32_t a = 1, b = 0;
        for (uint8_t c : raw) a = (a + c) % 65521, b = (b + a) % 65521;
        for (size_t at = 0; at < raw.size();) {
            const size_t n = std::min<size_t>(65535, raw.size() - at);
            z.push_back(at + n == raw.size() ? 1 : 0);
            z.push_back((uint8_t)n), z.push_back((uint8_t)(n >> 8));
            z.push_back((uint8_t)~n), z.push_back((uint8_t)(~n >> 8));
            z.insert(z.end(), raw.begin() + (long)at, raw.begin() + (long)(at + n));
            at += n;
        }
        const uint32_t adler = b << 16 | a;
        for (int i = 3; i >= 0; i--) z.push_back((uint8_t)(adler >> (i * 8)));

        // Encode before opening a file; only a completed private temporary replaces the destination.
        static std::atomic<uint64_t> next{0};
        std::string temporary;
        FILE* f = nullptr;
        for (int attempt = 0; attempt < 128; attempt++) {
            temporary = std::string(path) + ".part." + std::to_string(next.fetch_add(1, std::memory_order_relaxed));
            f = fopen(temporary.c_str(), "wbx");
            if (f || errno != EEXIST) break;
        }
        if (!f) return failed("open", errno);
        bool ok = true;
        unsigned error = 0;
        const char* phase = "write";
        auto put = [&](const void* bytes, size_t n) {
            if (ok && n && fwrite(bytes, 1, n, f) != n) {
                error = errno ? errno : EIO;
                ok = false;
            }
        };
        auto be32 = [&](uint32_t v) {
            const uint8_t q[4] = {(uint8_t)(v >> 24), (uint8_t)(v >> 16), (uint8_t)(v >> 8), (uint8_t)v};
            put(q, sizeof q);
        };
        auto chunk = [&](const char* type, const uint8_t* d, size_t n) {
            be32((uint32_t)n);
            put(type, 4);
            put(d, n);
            be32(crc32(d, n, crc32((const uint8_t*)type, 4)) ^ 0xFFFFFFFFu);
        };
        static const uint8_t sig[8] = {0x89, 'P', 'N', 'G', 13, 10, 26, 10};
        put(sig, sizeof sig);
        const uint8_t ihdr[13] = {(uint8_t)(w >> 24), (uint8_t)(w >> 16), (uint8_t)(w >> 8), (uint8_t)w,
                                 (uint8_t)(h >> 24), (uint8_t)(h >> 16), (uint8_t)(h >> 8), (uint8_t)h, 8, 2, 0, 0, 0};
        chunk("IHDR", ihdr, sizeof ihdr);
        chunk("IDAT", z.data(), z.size());
        chunk("IEND", nullptr, 0);
        if (fclose(f) != 0) {
            if (ok) phase = "close", error = errno ? errno : EIO;
            ok = false;
        }
        if (ok) {
            phase = "publish";
#ifdef _WIN32
            ok = MoveFileExA(temporary.c_str(), path, MOVEFILE_REPLACE_EXISTING) != 0;
            if (!ok) error = GetLastError();
#else
            ok = rename(temporary.c_str(), path) == 0;
            if (!ok) error = errno;
#endif
        }
        if (!ok) {
            if (remove(temporary.c_str()) != 0) failed("temporary cleanup", errno);
            return failed(phase, error);
        }
        return true;
    } catch (const std::exception&) {
        return failed("encoding allocation");
    }
}

namespace {

int64_t fps_n = 0;
std::chrono::steady_clock::time_point fps_t = std::chrono::steady_clock::now();

/// RX_VK_SLOW_LOG=<ms>: a frame that long after the last: its replay's times, for what the stutters are
void log_slow_frame() {
    static const int limit = getenv("RX_VK_SLOW_LOG") ? atoi(getenv("RX_VK_SLOW_LOG")) : 0;
    if (!limit) return;
    static auto last_t = std::chrono::steady_clock::now();
    static int64_t last[PROF_COUNT];
    static int64_t last_pl, last_tr, last_d, last_up;
    static int logged;
    const auto now = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(now - last_t).count();
    last_t = now;
    int64_t d = 0, sk = 0, di = 0, pl = 0, tr = 0, fl = 0, up = 0, astc = 0, upb = 0;
    draw_stats(&d, &sk, &di, &pl);
    shader_stats(&tr, &fl);
    texture_stats(&up, &astc, &upb);
    if (ms >= limit && d - last_d >= 500 && logged < 800) {
        logged++;
        const auto dp = [](int i) { return (prof_ns[i] - last[i]) / 1e6; };
        logf("vk: slow frame at present %lld: %.1f ms; submit %.1f, draw %.1f, descriptors %.1f, pipelines %.1f, programs %.1f, "
             "commands %.1f, memory sync %.1f, queue submit %.1f, slot wait %.1f; %lld draws, %lld new pipelines, %lld new programs, "
             "%lld uploads", (long long)presents, ms, dp(PROF_SUBMIT), dp(PROF_DRAW), dp(PROF_DESC), dp(PROF_PIPE), dp(PROF_PROGRAM),
             dp(PROF_CMDS), dp(PROF_MEMSYNC), dp(PROF_QSUBMIT), dp(PROF_SLOTWAIT), (long long)(d - last_d), (long long)(pl - last_pl),
             (long long)(tr - last_tr), (long long)(up - last_up));
    }
    for (int i = 0; i < PROF_COUNT; i++) last[i] = prof_ns[i];
    last_pl = pl, last_tr = tr, last_d = d, last_up = up;
}

uint64_t made_ring[32];
int64_t made_ring_present = 0;
void log_frame_stats() {
    log_slow_frame();
    made_ring[presents % 32] = texture_made();
    made_ring_present = presents;
    if (++fps_n == 120) {
        const auto now = std::chrono::steady_clock::now();
        const double secs = std::chrono::duration<double>(now - fps_t).count();
        int64_t d = 0, sk = 0, di = 0, pl = 0, tr = 0, fl = 0, passes = 0, mesh = 0, up = 0, astc = 0, upb = 0;
        int64_t st_clears = 0, masked = 0;
        clear_stats(&st_clears, &masked);
        double pms = 0;
        draw_stats(&d, &sk, &di, &pl);
        draw_stats2(&passes, &mesh, &pms);
        shader_stats(&tr, &fl);
        texture_stats(&up, &astc, &upb);
        int64_t pskip = 0, pwaits = 0;
        double pwait_ms = 0;
        draw_stats3(&pskip, &pwaits, &pwait_ms);
        static int64_t l_pskip, l_pwaits;
        static double l_pwait_ms;
        static int64_t l_reused, l_saves;
        int64_t links = 0, lib_hits = 0, lib_cold = 0, lib_built = 0;
        double link_ms = 0, lib_build_ms = 0;
        static int64_t l_links;
        static double l_link_ms;
        char libs[256] = "";   // (the stage libraries' part of the line, with RX_VK_GPL only: other logs keep the line they had)
        if (draw_stats4(&links, &link_ms, &lib_hits, &lib_cold, &lib_built, &lib_build_ms))
            snprintf(libs, sizeof libs, "; %.2f library links tried (%.2f ms), %lld libraries the driver had and %lld cold in all; "
                     "%lld of the cold built by workers (%.0f ms)", (links - l_links) / 120.0, (link_ms - l_link_ms) / 120.0,
                     (long long)lib_hits, (long long)lib_cold, (long long)lib_built, lib_build_ms);
        logf("vk: a frame: %.0f draws kept their neighbour's descriptors (%.0f of them through other slots' registrations)",
             (draw_desc_reused() - l_reused) / 120.0, (draw_slotgen_saves() - l_saves) / 120.0);
        l_reused = draw_desc_reused(), l_saves = draw_slotgen_saves();
        logf("vk: a frame: %.1f draws left out (not ready), %.2f waits for shaders or pipelines (%.1f ms); "
             "translation %.0f ms in all, %d jobs queued%s",
             (pskip - l_pskip) / 120.0, (pwaits - l_pwaits) / 120.0, (pwait_ms - l_pwait_ms) / 120.0,
             shader_translate_ms(), jobs_pending(), libs);
        l_pskip = pskip, l_pwaits = pwaits, l_pwait_ms = pwait_ms, l_links = links, l_link_ms = link_ms;
        {
            static const char* names[PROF_COUNT] = {"submit", "present", "draw", "descriptors", "texture lookups",
                                                    "uploads", "pipelines", "programs", "commands", "dispatch",
                                                    "memory sync", "queue submit", "slot wait"};
            static int64_t last[PROF_COUNT];
            char line[600];
            int n = snprintf(line, sizeof line, "vk: a frame's replay time (ms):");
            for (int i = 0; i < PROF_COUNT; i++) {
                n += snprintf(line + n, sizeof line - n, " %s %.2f%s", names[i], (prof_ns[i] - last[i]) / 120.0 / 1e6,
                              i + 1 < PROF_COUNT ? "," : "");
                last[i] = prof_ns[i];
            }
            logf("%s", line);
        }
        int64_t mirrors = 0, sync_b = 0, sync_c = 0, wb_b = 0;
        uint64_t mirror_mb = 0;
        memory_stats(&mirrors, &mirror_mb, &sync_b, &sync_c, &wb_b);
        static int64_t l_sync_b, l_sync_c, l_wb_b;
        logf("vk: a frame: mirrors %lld (%llu MB): %.2f MB synced in %.1f copies, %.2f MB written back",
             (long long)mirrors, (unsigned long long)mirror_mb, (sync_b - l_sync_b) / 120.0 / 1048576.0,
             (sync_c - l_sync_c) / 120.0, (wb_b - l_wb_b) / 120.0 / 1048576.0);
        l_sync_b = sync_b, l_sync_c = sync_c, l_wb_b = wb_b;
        static int64_t l_d, l_sk, l_di, l_pass, l_sub, l_up, l_upb, l_bar, l_ubu, l_ubo, l_wait;
        logf("vk: %lld presents, %.1f fps; a frame: %.0f draws (%.0f skipped), %.0f dispatches, %.0f passes, %.0f "
             "barriers, %.0f submits, %.1f uploads (%.1f MB), %.0f ub updates (%.0f overlays), GPU-bound wait %.1f ms; "
             "%lld pipelines (%.0f ms), %lld programs (%lld failed), %zu textures (%llu MB)",
             (long long)presents, 120.0 / secs, (d - l_d) / 120.0, (sk - l_sk) / 120.0, (di - l_di) / 120.0,
             (passes - l_pass) / 120.0, (st_barriers - l_bar) / 120.0, (st_submits - l_sub) / 120.0,
             (up - l_up) / 120.0, (upb - l_upb) / 120.0 / 1048576.0, (st_ub_updates - l_ubu) / 120.0,
             (st_ub_overlays - l_ubo) / 120.0, (st_wait_ns - l_wait) / 120.0 / 1e6, (long long)pl, pms,
             (long long)tr, (long long)fl, texture_count(), (unsigned long long)(texture_bytes() >> 20));
        draw_skip_report(120.0);
        {   // (hookshot: the full barriers recorded and the ones left out as the second of two in a row, RX_VK_BARRIER_DEDUPE_OFF=1)
            static int64_t l_made, l_left;
            int64_t made = 0, left = 0;
            barrier_stats(&made, &left);
            static int64_t l_relaxed;
            const int64_t relaxed = pass_barriers_relaxed();
            logf("vk: a frame: %.1f full barriers recorded, %.1f left out (nothing recorded since the one before), %.1f pass starts "
                 "ordered in their fragment stages only (RX_VK_PASS_BARRIER=game)", (made - l_made) / 120.0, (left - l_left) / 120.0,
                 (relaxed - l_relaxed) / 120.0);
            l_made = made, l_left = left, l_relaxed = relaxed;
            std::string fl;   // (NVN's barrier bits: ORDER_PRIMITIVES 1, ORDER_FRAGMENTS 2, ORDER_FRAGMENTS_TILED 4, ORDER_INDIRECT_DATA 8, INVALIDATE_TEXTURE 0x10, _SHADER 0x20, _TEXTURE_DESCRIPTOR 0x40, _ZCULL 0x80)
            std::lock_guard lk(st_barrier_flags_mu);
            for (const auto& [bits, n] : st_barrier_flags) {
                char b[48];
                snprintf(b, sizeof b, " %#x: %.1f;", (unsigned)bits, n / 120.0);
                fl += b;
            }
            logf("vk: a frame: the game's barriers by their NVN flags:%s", fl.c_str());
            st_barrier_flags.clear();
        }
        static int64_t l_gpu;
        static uint64_t l_made;
        logf("vk: GPU time %.1f ms a frame (the longest submission %.1f ms); %.1f textures made a frame",
             (st_gpu_ns - l_gpu) / 120.0 / 1e6, st_gpu_max_ns / 1e6, (texture_made() - l_made) / 120.0);
        {   // (issue 109: the movie networks too slow on this card: left out from now on, vk_draw.cpp movie_networks_slow)
            static const bool keep_all = getenv("RX_VK_NETWORK_KEEP") && !strcmp(getenv("RX_VK_NETWORK_KEEP"), "all");
            const double gpu_ms = (st_gpu_ns - l_gpu) / 120.0 / 1e6, draws = (d - l_d) / 120.0;
            if (!keep_all && gpu_ms > 40.0 && draws < 40.0 && !movie_networks_slow.exchange(true))
                logf("vk: %.0f ms of GPU time a frame for %.0f draws (a movie): its upscaler networks are left out from now on "
                     "(a linear upscale; RX_VK_NETWORK_KEEP=all keeps them)", gpu_ms, draws);
        }
        l_made = texture_made();
        {
            static int64_t l_ub, l_un;
            int64_t ub = 0, un = 0;
            texture_unswizzle_stats(&ub, &un);
            logf("vk: a frame: %.2f MB unswizzled on the CPU in %.2f ms", (ub - l_ub) / 120.0 / 1048576.0,
                 (un - l_un) / 120.0 / 1e6);
            l_ub = ub, l_un = un;
        }
        l_gpu = st_gpu_ns, st_gpu_max_ns = 0;
        gprof_report(120);
        l_d = d, l_sk = sk, l_di = di, l_pass = passes, l_sub = st_submits, l_up = up, l_upb = upb, l_bar = st_barriers;
        l_ubu = st_ub_updates, l_ubo = st_ub_overlays, l_wait = st_wait_ns;
        fps_t = now, fps_n = 0;
    }
}

/// (hookshot) Once a game frame: every 600 a status line (the video memory, DLSS's details while it draws), and every 60
/// a look at the driver's allowance, warned of once when the use passes it (again only after it was back under)
void status_tick() {
    namespace sl = status_line;
    static uint64_t frames = 0;
    // (the slowest frame since the last line, present to present: a stall shows here, to be matched with what the log says
    // around it, such as DLSS's feature made again)
    static int64_t last_ns = 0, slowest_ns = 0;
    const int64_t now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    if (last_ns) slowest_ns = std::max(slowest_ns, now_ns - last_ns);
    // (RX_VK_FRAME_TRACE=1: every frame a line of hookshot_frames.csv beside the game, present to present and what the frame did,
    // so that each stutter can be matched to its cause: waits for shaders or pipelines, translation, uploads, the memory sync,
    // pipeline compiles, the GPU behind. RX_VK_FRAME_LOG has the frame times alone)
    static FILE* trace = [] {
        if (!getenv("RX_VK_FRAME_TRACE") || !atoi(getenv("RX_VK_FRAME_TRACE"))) return (FILE*)nullptr;
        char exe[MAX_PATH];
        const DWORD n = game_module_path(exe, MAX_PATH);
        std::string dir(exe, n);
        FILE* f = fopen((dir.substr(0, dir.find_last_of("\\/") + 1) + "hookshot_frames.csv").c_str(), "w");
        if (f) fprintf(f, "frame,ms,pipe_waits,pipe_wait_ms,left_out,translate_ms,uploads,upload_mb,textures,memsync_ms,pipelines_ms,"
                          "uploads_ms,programs_ms,slot_wait_ms,gpu_behind_ms,draw_ms,desc_ms,cmds_ms,tex_lookup_ms,textures_gone\n");
        return f;
    }();
    if (trace && last_ns) {
        static int64_t l_ps, l_pw, l_up, l_upb, l_wait, l_prof[PROF_COUNT];
        static double l_pwms, l_tr;
        int64_t ps = 0, pw = 0, up = 0, astc = 0, upb = 0;
        double pwms = 0;
        draw_stats3(&ps, &pw, &pwms);
        texture_stats(&up, &astc, &upb);
        const double tr = shader_translate_ms();
        const auto dp = [](int i) { return (prof_ns[i] - l_prof[i]) / 1e6; };
        static uint64_t l_gone;
        const uint64_t gone = texture_destroy_generation();
        fprintf(trace, "%llu,%.2f,%lld,%.2f,%lld,%.1f,%lld,%.2f,%zu,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%llu\n",
                (unsigned long long)frames, (now_ns - last_ns) / 1e6, (long long)(pw - l_pw), pwms - l_pwms, (long long)(ps - l_ps),
                tr - l_tr, (long long)(up - l_up), (upb - l_upb) / 1048576.0, texture_count(), dp(PROF_MEMSYNC), dp(PROF_PIPE),
                dp(PROF_UPLOAD), dp(PROF_PROGRAM), dp(PROF_SLOTWAIT), (st_wait_ns - l_wait) / 1e6, dp(PROF_DRAW), dp(PROF_DESC),
                dp(PROF_CMDS), dp(PROF_TEXGET), (unsigned long long)(gone - l_gone));
        l_gone = gone;
        if (frames % 60 == 0) fflush(trace);
        l_ps = ps, l_pw = pw, l_pwms = pwms, l_up = up, l_upb = upb, l_tr = tr, l_wait = st_wait_ns;
        std::memcpy(l_prof, prof_ns, sizeof l_prof);
    }
    last_ns = now_ns;
    if (++frames % sl::kCheckEvery) return;
    uint64_t used = 0, allowed = 0;
    const bool known = vram_usage_now(&used, &allowed);
    static bool told_unknown = false;
    if (!known && !told_unknown) {
        told_unknown = true;
        logf("vk: status: the driver gives no video-memory budget (VK_EXT_memory_budget): only the game's own allocations are known");
    }
    char dlss[320] = "";
    const bool line = frames % sl::kEvery == 0;
    const int mode = line ? temporal_status(dlss, sizeof dlss) : temporal_mode();   // (temporal_status: since the last line, so only for one)
    static bool over = false;
    if (known && sl::crossed(over, used, allowed)) {
        char w[200];
        sl::over_text(w, sizeof w, used, allowed, mode);
        logf("%s", w);
    }
    if (!line) return;
    char v[128];
    sl::vram_text(v, sizeof v, used, allowed, known);
    logf("vk: status: %d frames, the slowest %.1f ms; %s%s", sl::kEvery, slowest_ns / 1e6, v, dlss);
    slowest_ns = 0;
}

std::atomic<bool> present_native_pq{false};   // (do_present: the game's picture is already HDR10)
uint64_t present_value = 0;   // (the present job's value, handed out as it came; 0: the next one)
struct PendingShot {
    bool on = false;
    uint64_t value = 0;
    int64_t presents = 0;
    std::shared_ptr<std::vector<uint8_t>> pixels;
};
std::shared_ptr<std::vector<uint8_t>> shot_pixels(uint64_t at_value, int64_t at_present) {
    wait_value(at_value);
    // The completed transfer may target non-coherent host memory.
    invalidate_mapped(shot_allocs[at_present & 1], (uint64_t)kShotW * kShotH * 4);
    const uint8_t* const sm = (const uint8_t*)shot_maps[at_present & 1];
    return std::make_shared<std::vector<uint8_t>>(sm, sm + (size_t)kShotW * kShotH * 4);
}
void do_present(const nvn_tex* t, const int crop[4]) {
    static PendingShot pending;
    // (hookshot) NVIDIA Reflex (vk_reflex.cpp): this frame's simulation ended, its submission starts; on the way out (after the
    // present) the driver's sleep until the next frame should begin. AMD Anti-Lag 2 (vk_antilag.cpp) waits there as well: the
    // next frame's input stage
    temporal_idle();   // (DLSS, FSR: their video memory let go once not used for a while)
    status_tick();   // (the status line, and the video-memory warning)
    reflex_present_begins();
    struct ReflexSleep { ~ReflexSleep() { reflex_sleep(); antilag_input(); } } reflex_sleep_after;
    {   // (hookshot, RX_VK_PASS_TRACE_AT=<frames>: the passes of the frames around these frame numbers, for finding what a frame is made of)
        static std::atomic<int64_t> seen{0};
        static const int64_t at = getenv("RX_VK_PASS_TRACE_AT") ? atoll(getenv("RX_VK_PASS_TRACE_AT")) : 0;
        if (at && ++seen % at == 0) pass_traces_log(texture_frame_no() - 1, texture_frame_no());
    }
    static const char* shots = getenv("RX_VK_SHOTS");
    static const int shot_every = getenv("RX_VK_SHOT_EVERY") ? std::max(1, atoi(getenv("RX_VK_SHOT_EVERY"))) : 300;
    VkImage src = VK_NULL_HANDLE;
    VkImageView direct_view = VK_NULL_HANDLE;
    uint32_t sw = 0, sh = 0;
    bool shot = false, interpolated = false;
    VkImage made_src = VK_NULL_HANDLE;
    {
        std::lock_guard lk(replay_mu);
        bench_frame(presents);
        SlowWatch slow("present");
        const bool stereo_frame = stereo_held;
        frame_submits.clear();
        PROF(PROF_PRESENT);
        if (stereo_frame) {
            stereo_held = false;
            if (t && !dev.offscreen)
                if (Texture* vt = texture_for_present(t); vt && vt->gpu_written) vr::stereo_present(vt->image, vt->width, vt->height, crop);
            vr::stereo_frame_done();
        }
        const uint64_t value = present_value ? present_value : ++q_values;
        present_value = 0;
        rec_value_now = value;
        ub_first_of_recording = ub_updates.size();
        Recording& r = begin_recording(value);
        memory_begin_submission();
        Texture* tex = t ? texture_for_present(t) : nullptr;
        static int logged_p;
        if (logged_p++ < 4)
            logf("vk: present texture %#llx %ux%u format %#x: %s", (unsigned long long)(t ? t->gpu : 0),
                 t ? t->d.width : 0, t ? t->d.height : 0, t ? t->d.format : 0,
                 tex ? (tex->gpu_written ? "rendered" : "never rendered") : "no image");
        if (tex) src = tex->image, sw = tex->width, sh = tex->height;
        // (the Insert menu's HDR set to Native: the game, told its display is HDR10, presents PQ codes in RGB10A2 (vk
        // A2B10G10R10); that picture goes to the HDR10 swapchain as it is, not through the SDR-to-HDR pass)
        present_native_pq.store(tex && dev.hdr && opts.hdr == 2 && tex->format == VK_FORMAT_A2B10G10R10_UNORM_PACK32,
                                std::memory_order_relaxed);
        if (tex && !dev.offscreen && (opts.upscale || opts.sharpness || dev.hdr || wide_sides_possible())) {   // (and ultrawide's menu sides)
            // Keep the sampled view with the same cached image as the direct present, before advancing its age.
            direct_view = texture_view(tex, t, nullptr, VK_IMAGE_VIEW_TYPE_2D, ViewUse::Sample);
            tex->last_frame = texture_frame_no();
            tex->last_value = std::max(tex->last_value, value);
        }
        // (frame interpolation: made frames and this one shown by the display thread, vk_framegen.cpp. Recorded
        // after the submit that signals the present's value, in a submit of its own: the game waits for that value
        // before it submits its next frame, and the interpolation's GPU work (~1 ms) held it back. The display
        // thread's present follows it on the one queue. RX_VK_SHOT_MADE shots need it in this recording.)
        static const bool fg_inline = getenv("RX_VK_SHOT_MADE") != nullptr;
        const bool fg_after = framegen_on() && tex && !fg_inline;
        interpolated = framegen_on() && tex && fg_inline &&
                       framegen_record(r.cmd, texture_view(tex, t, nullptr, VK_IMAGE_VIEW_TYPE_2D, ViewUse::Sample), sw,
                                       sh, crop);
        texture_frame(completed());
        full_barrier(r.cmd);
        presents++;
        presents_now = presents;
        wide_sides_tick();
        if (presents == 1) hang_watch_start();
        {   // (EvilBoyce, diagnostics) RX_VK_WATCH_CPU=<present>:<gpu address>[,...] (hex addresses): from that present on, the
            // CPU's writes to those 4 KB pages logged with the writer's stack (host page watch: 12 reports)
            static bool armed = false;
            static const char* e = getenv("RX_VK_WATCH_CPU");
            if (e && !armed && host->watch_guest_writes && presents >= atoll(e)) {
                armed = true;
                for (const char* q = strchr(e, ':'); q && *q;) {
                    char* end;
                    const uint64_t a = strtoull(q + 1, &end, 16);
                    if (end == q + 1) break;
                    if (const uint64_t g = host->gpu_guest(a)) host->watch_guest_writes(g & ~4095ull, 4096);
                    q = *end ? end : nullptr;
                }
            }
        }
        ts_anchor_ns = host->ticks() * 32;
        dump_present(presents);
        if (src && (dev.offscreen || shots) && offscreen_init()) {
            if (dev.offscreen) {
                image_barrier(r.cmd, off_image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
                blit_crop(r.cmd, src, sw, sh, crop, off_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          dev.swap_extent.width, dev.swap_extent.height);
            }
            // (RX_VK_SHOT_FROM / _TO: every present in that range too)
            static const int64_t shot_from = getenv("RX_VK_SHOT_FROM") ? atoll(getenv("RX_VK_SHOT_FROM")) : -1;
            static const int64_t shot_to = getenv("RX_VK_SHOT_TO") ? atoll(getenv("RX_VK_SHOT_TO")) : -1;
            shot = shots && (presents % shot_every == 0 || (presents >= shot_from && presents <= shot_to));
            // (RX_VK_SHOT_GO=<file>: every present while that file is there too, a test's run of frames)
            static const char* shot_go = getenv("RX_VK_SHOT_GO");
            if (shots && shot_go && !shot) {
                struct stat sb;
                shot = stat(shot_go, &sb) == 0;
            }
            // (RX_VK_SHOT_MENU=1, a debug aid: without a window the Insert menu is neither ticked nor drawn; this ticks it
            // at each present and draws it into the shots, for pictures of it from a headless run)
            static const bool shot_menu = getenv("RX_VK_SHOT_MENU") != nullptr;
            if (shot_menu && dev.offscreen) menu_tick();
            // (RX_VK_SHOT_MADE=1: the frame interpolation made before it too, written beside it as vk_N_made.png)
            static const bool shot_made = getenv("RX_VK_SHOT_MADE") != nullptr;
            uint32_t mw = 0, mh = 0;
            made_src = shot && shot_made && interpolated ? framegen_made_image(&mw, &mh) : VK_NULL_HANDLE;
            if (made_src && !made_shot_init()) made_src = VK_NULL_HANDLE;
            if (made_src) {
                image_barrier(r.cmd, made_image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
                blit_crop(r.cmd, made_src, mw, mh, nullptr, made_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, kShotW, kShotH);
                image_barrier(r.cmd, made_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
                const VkBufferImageCopy c{0, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {kShotW, kShotH, 1}};
                vkCmdCopyImageToBuffer(r.cmd, made_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, made_buf, 1, &c);
                capture_host_read_barrier(r.cmd, made_buf, (VkDeviceSize)kShotW * kShotH * 4);
            }
            if (shot) {
                // A pending shot may be two (or more) presents old. Own its pixels before this slot is written again.
                if (pending.on && (pending.presents & 1) == (presents & 1) && !pending.pixels)
                    pending.pixels = shot_pixels(pending.value, pending.presents);
                image_barrier(r.cmd, shot_image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
                // (RX_VK_SHOT_FX=1: through the window's scale pass too, ultrawide's menu sides as the window shows them. For headless
                // runs: in a window it resizes the pass's image to the shot's size and the present back again, two device waits a
                // shot, and it is not under present_image_mu, which frame generation's display thread takes for the same pass)
                static const bool shot_fx = getenv("RX_VK_SHOT_FX") != nullptr;
                const VkImage fx_img = shot_fx && tex ? framegen_scale(r.cmd, texture_view(tex, t, nullptr, VK_IMAGE_VIEW_TYPE_2D, ViewUse::Sample),
                                                                       src, sw, sh, crop, kShotW, kShotH) : VK_NULL_HANDLE;
                if (fx_img) full_barrier(r.cmd), blit_crop(r.cmd, fx_img, kShotW, kShotH, nullptr, shot_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, kShotW, kShotH);
                else blit_crop(r.cmd, src, sw, sh, crop, shot_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, kShotW, kShotH);
                if (shot_menu && dev.offscreen) menu_record_image(r.cmd, shot_image, shot_view, kShotW, kShotH);   // (headless: the menu over the shot)
                image_barrier(r.cmd, shot_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
                const VkBufferImageCopy c{0, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {kShotW, kShotH, 1}};
                vkCmdCopyImageToBuffer(r.cmd, shot_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, shot_bufs[presents & 1], 1, &c);
                capture_host_read_barrier(r.cmd, shot_bufs[presents & 1], (VkDeviceSize)kShotW * kShotH * 4);
            }
        }
        memory_end_submission(r.cmd);
        full_barrier(r.cmd);
        submit_recording(r);   // (nothing is stopped at a present: drain)
        if (pushed.load() < value) pushed = value;
        done_max = std::max(done_max, value), signaled = std::max(signaled, value);
        if (fg_after) {
            // (its slot is free again once the next submit's value is reached: that one comes after it)
            Recording& g = begin_recording(value + 1);
            interpolated = framegen_record(g.cmd, texture_view(tex, t, nullptr, VK_IMAGE_VIEW_TYPE_2D, ViewUse::Sample),
                                           sw, sh, crop);
            submit_recording(g, false);
        }
        if (interpolated) framegen_queue(value);
        else menu_note_frame(true);   // (the FPS counter's, as frame generation's thread counts what it shows)
        {   // (RX_VK_WAIT_GPU=<frames>: the replay no further ahead of the GPU than that)
            static uint64_t values[8];
            static const int ahead = [] {
                const char* e = getenv("RX_VK_WAIT_GPU");
                const long count = e ? std::strtol(e, nullptr, 10) : 0;
                constexpr long capacity = sizeof(values) / sizeof(values[0]);
                if (count > capacity) logf("vk: RX_VK_WAIT_GPU=%ld exceeds the ring capacity; using %ld", count, capacity);
                return (int)std::clamp<long>(count, 0, capacity);
            }();
            if (ahead > 0) {
                const uint64_t old = values[presents % ahead];
                values[presents % ahead] = value;
                if (old) wait_value(old);
            }
        }
        if (presents % 300 == 0) pipeline_cache_save(false);
        {   // (RX_VK_DUMP_KEY: F9 dumps the next frame, 8 times at most)
            static const bool dump_key = getenv("RX_VK_DUMP_KEY") != nullptr;
            static bool was_down;
            static int key_left = 8;
            const bool down = dump_key && (GetAsyncKeyState(VK_F9) & 0x8000) != 0;
            if (down && !was_down && key_left > 0) {
                key_left--;
                static const int n = getenv("RX_VK_DUMP_KEY_N") ? std::max(1, atoi(getenv("RX_VK_DUMP_KEY_N"))) : 1;
                logf("vk: F9 at present %lld: the next %d frame(s) are dumped", (long long)presents, n);
                dump_request(presents + 1, n);
            }
            was_down = down;
        }
        {   // (RX_VK_DUMP_FILE=<path>: that file appearing dumps the next frame, and is deleted: a test that steers the
            // game live (HOS_INPUT_LIVE) dumps where it stands)
            // (RX_VK_DUMP_MOD=<m>,<r>[,<n>]: the file arms the next n frames (default 2) whose present is r mod m
            // instead: a glitch that comes back every m frames, dumped in its own frame (the flash detector names it))
            static const char* dump_file = getenv("RX_VK_DUMP_FILE");
            static int mod_m = -1, mod_r = 0, mod_n = 2, mod_left = 0;
            if (mod_m < 0) {
                mod_m = 0;
                if (const char* e = getenv("RX_VK_DUMP_MOD")) sscanf(e, "%d,%d,%d", &mod_m, &mod_r, &mod_n);
            }
            if (dump_file && presents % 10 == 0 && GetFileAttributesA(dump_file) != INVALID_FILE_ATTRIBUTES) {
                DeleteFileA(dump_file);
                if (mod_m > 0) {
                    mod_left = std::max(1, mod_n);
                    logf("vk: dump file at present %lld: the next %d frames at %d mod %d are dumped", (long long)presents,
                         mod_left, mod_r, mod_m);
                } else {
                    logf("vk: dump file at present %lld: the next frame is dumped", (long long)presents);
                    dump_request(presents + 1, 1);
                }
            }
            if (mod_left > 0 && (presents + 2) % mod_m == mod_r) {   // (armed at the present before the frame's own)
                mod_left--;
                dump_request(presents + 1, 1);
            }
        }
        // (a shot is read once the GPU reached its frame, at a later present, and written on a thread of its own:
        // waiting for the GPU and encoding the PNG there took the replay ~570 ms a shot, and frequent shots
        // halved the frame rate test runs measured. RX_VK_SHOT_MADE shots are taken at once, as before.)
        auto take = [&](uint64_t at_value, int64_t at_present) {
            auto px = pending.on && pending.presents == at_present && pending.pixels
                          ? std::move(pending.pixels) : shot_pixels(at_value, at_present);
            // (RX_VK_DUMP_BLACK=<n>: a shot with 12-90% pure black pixels in a lit picture, or a green cast over it,
            // dumps the next frame, n times at most: the glitches some scenes show now and then)
            static int black_left = getenv("RX_VK_DUMP_BLACK") ? atoi(getenv("RX_VK_DUMP_BLACK")) : 0;
            if (black_left > 0) {
                const uint8_t* q0 = px->data();
                uint64_t black = 0, white = 0, sum = 0, ch[3] = {};
                const uint64_t n = (uint64_t)kShotW * kShotH;
                for (uint64_t i = 0; i < n; i++) {
                    const uint8_t* q = q0 + i * 4;
                    black += q[0] < 3 && q[1] < 3 && q[2] < 3;
                    white += q[0] > 235 && q[1] > 235 && q[2] > 235;
                    sum += q[0] + q[1] + q[2];
                    ch[0] += q[0], ch[1] += q[1], ch[2] += q[2];
                }
                // (hookshot: a picture blown out to white, the HUD on it: written as white_<present>.png, 40 of them)
                static int white_left = 40;
                if (at_present > 600 && white * 100 > n * 40 && white_left > 0) {
                    white_left--;
                    logf("vk: present %lld: %.1f%% white pixels, mean %llu %llu %llu", (long long)at_present, 100.0 * white / n,
                         (unsigned long long)(ch[0] / n), (unsigned long long)(ch[1] / n), (unsigned long long)(ch[2] / n));
                    auto w0 = px;
                    const std::string name = std::string(shots) + "/white_" + std::to_string(at_present) + ".png";
                    std::thread([w0, name] { write_png(name.c_str(), w0->data(), kShotW, kShotH); }).detach();
                }
                const bool green = ch[1] * 10 > ch[0] * 12 && ch[1] * 10 > ch[2] * 14 && ch[1] / n > 120;
                if (at_present > 600 && ((black * 8 > n && black * 100 < n * 97 && sum / (n * 3) > 4) || green)) {
                    black_left--;
                    logf("vk: present %lld: %.1f%% black pixels, mean %llu %llu %llu: the next frame is dumped",
                         (long long)at_present, 100.0 * black / n, (unsigned long long)(ch[0] / n),
                         (unsigned long long)(ch[1] / n), (unsigned long long)(ch[2] / n));
                    dump_request(presents + 1);
                }
            }
            // (RX_VK_FLASH=1, with RX_VK_SHOT_EVERY=1: a frame that differs from both its neighbours in a part of the
            // picture where they agree is a one-frame flash: the three are written as flash_<present>_{a,b,c}.png and
            // the next frame is dumped (a few times). The other shots are written every 300th present only.)
            static const bool flash = getenv("RX_VK_FLASH") && atoi(getenv("RX_VK_FLASH")) != 0;
            if (flash) {
                constexpr uint32_t W = 64, H = 36;
                auto small = std::make_shared<std::vector<uint8_t>>(W * H * 3);
                for (uint32_t y = 0; y < H; y++) {
                    const uint32_t y0 = (uint64_t)y * kShotH / H;
                    const uint32_t y1 = std::max(y0 + 1, (uint32_t)((uint64_t)(y + 1) * kShotH / H));
                    for (uint32_t x = 0; x < W; x++) {
                        const uint32_t x0 = (uint64_t)x * kShotW / W;
                        const uint32_t x1 = std::max(x0 + 1, (uint32_t)((uint64_t)(x + 1) * kShotW / W));
                        for (int ch = 0; ch < 3; ch++) {
                            uint64_t sum = 0, samples = 0;
                            // Partition the full image and count the samples actually visited. A partial
                            // stride must not brighten the cell or wrap its mean back through uint8.
                            for (uint32_t yy = y0; yy < y1; yy += 4)
                                for (uint32_t xx = x0; xx < x1; xx += 4) {
                                    sum += (*px)[((size_t)yy * kShotW + xx) * 4 + ch];
                                    samples++;
                                }
                            (*small)[(y * W + x) * 3 + ch] = (uint8_t)(sum / samples);
                        }
                    }
                }
                static std::shared_ptr<std::vector<uint8_t>> sa, sb, pa, pb;
                static int64_t present_b = -1;
                static int flashes_left = 400, dumps_left = getenv("RX_VK_FLASH_DUMPS") ? atoi(getenv("RX_VK_FLASH_DUMPS")) : 6;
                if (sa && sb && present_b == at_present - 1) {
                    // (RX_VK_FLASH_DIFF, RX_VK_FLASH_CELLS: a subtler flash; logged with how many cells went up, and where)
                    static const int fl_diff = getenv("RX_VK_FLASH_DIFF") ? atoi(getenv("RX_VK_FLASH_DIFF")) : 48;
                    static const int fl_cells = getenv("RX_VK_FLASH_CELLS") ? atoi(getenv("RX_VK_FLASH_CELLS")) : 40;
                    int hits = 0, up = 0, minx = W, maxx = -1, miny = H, maxy = -1, peak = 0;
                    uint64_t tab = 0, tac = 0;   // (camera motion: a and c differ as much as either from b; a flash: a ~ c)
                    for (uint32_t i = 0; i < W * H * 3; i++)
                        tab += std::abs((*sa)[i] - (*sb)[i]), tac += std::abs((*sa)[i] - (*small)[i]);
                    for (uint32_t i = 0; i < W * H; i++) {
                        int dab = 0, dbc = 0, dac = 0;
                        for (int ch = 0; ch < 3; ch++) {
                            dab = std::max(dab, std::abs((*sa)[i * 3 + ch] - (*sb)[i * 3 + ch]));
                            dbc = std::max(dbc, std::abs((*sb)[i * 3 + ch] - (*small)[i * 3 + ch]));
                            dac = std::max(dac, std::abs((*sa)[i * 3 + ch] - (*small)[i * 3 + ch]));
                        }
                        if (dab > fl_diff && dbc > fl_diff && dac < fl_diff / 3) {
                            hits++;
                            up += (*sb)[i * 3 + 1] > (*sa)[i * 3 + 1];
                            peak = std::max(peak, dab);
                            const int cx = (int)(i % W), cy = (int)(i / W);
                            minx = std::min(minx, cx), maxx = std::max(maxx, cx);
                            miny = std::min(miny, cy), maxy = std::max(maxy, cy);
                        }
                    }
                    // (RX_VK_FLASH_REGION=<level>: also while the picture moves, when the cells' test cannot hold: a 16x9 block of
                    // the picture whose mean brightness jumps by the level and comes back the next frame)
                    static const int fl_region = getenv("RX_VK_FLASH_REGION") ? atoi(getenv("RX_VK_FLASH_REGION")) : 0;
                    int region_hits = 0, region_peak = 0, region_at = -1;
                    if (fl_region)
                        for (uint32_t ry = 0; ry < 9; ry++)
                            for (uint32_t rx = 0; rx < 16; rx++) {
                                int m[3] = {0, 0, 0};
                                for (uint32_t y = ry * 4; y < ry * 4 + 4; y++)
                                    for (uint32_t x = rx * 4; x < rx * 4 + 4; x++) {
                                        const size_t o = (y * W + x) * 3 + 1;
                                        m[0] += (*sa)[o], m[1] += (*sb)[o], m[2] += (*small)[o];
                                    }
                                const int ab = (m[1] - m[0]) / 16, cb = (m[1] - m[2]) / 16, ac = std::abs(m[0] - m[2]) / 16;
                                if (((ab > fl_region && cb > fl_region) || (ab < -fl_region && cb < -fl_region)) &&
                                    ac * 2 < std::min(std::abs(ab), std::abs(cb))) {
                                    region_hits++;
                                    if (std::abs(ab) > std::abs(region_peak)) region_peak = ab, region_at = (int)(ry * 16 + rx);
                                }
                            }
                    if (region_hits && flashes_left > 0)
                        logf("vk: region flash at present %lld: %d regions, peak %+d at region %d,%d", (long long)present_b, region_hits,
                             region_peak, region_at % 16, region_at / 16);
                    if (((hits >= fl_cells && tac * 2 < tab) || region_hits) && flashes_left > 0) {
                        flashes_left--;
                        logf("vk: flash at present %lld: %d of %u cells differ from both neighbours, %d brighter, peak %d, "
                             "cells %d,%d-%d,%d (now present %lld, texture frame %llu)", (long long)present_b, hits, W * H, up,
                             peak, minx, miny, maxx, maxy, (long long)presents, (unsigned long long)texture_frame_no());
                        const std::string base = std::string(shots) + "/flash_" + std::to_string(present_b);
                        auto a0 = pa, b0 = pb, c0 = px;
                        std::thread([a0, b0, c0, base] {
                            write_png((base + "_a.png").c_str(), a0->data(), kShotW, kShotH);
                            write_png((base + "_b.png").c_str(), b0->data(), kShotW, kShotH);
                            write_png((base + "_c.png").c_str(), c0->data(), kShotW, kShotH);
                        }).detach();
                        if (dumps_left > 0) dumps_left--, dump_request(presents + 1);
                        static int records_left = 12;   // (RX_VK_FLASH_RECORD: the draws around it)
                        const uint64_t tf = (uint64_t)(present_b + (int64_t)texture_frame_no() - presents);
                        probe_log(tf);
                        if (records_left > 0) records_left--, draw_records_log(tf - 1, tf + 1), pass_traces_log(tf - 1, tf + 1);
                    }
                }
                sa = sb, pa = pb, sb = small, pb = px, present_b = at_present;
                if (at_present % 300) return;
            }
            std::string path = std::string(shots) + "/vk_" + std::to_string(1000000 + at_present).substr(1) + ".png";
            std::thread([px, path, at_present] {
                if (write_png(path.c_str(), px->data(), kShotW, kShotH))
                    logf("vk: present %lld written to %s", (long long)at_present, path.c_str());
            }).detach();
        };
        if (pending.on && (shot || completed() >= pending.value)) take(pending.value, pending.presents), pending.on = false;
        if (shot && made_src && made_map) {
            take(value, presents);
            invalidate_mapped(made_buf_alloc, (uint64_t)kShotW * kShotH * 4);
            char path[1024];
            snprintf(path, sizeof path, "%s/vk_%06lld_made.png", shots, (long long)presents);
            write_png(path, (const uint8_t*)made_map, kShotW, kShotH);
        } else if (shot) {
            pending = {true, value, presents, {}};
        }
        log_frame_stats();
    }
    if (dev.offscreen || interpolated) return;
    present_image(src, sw, sh, crop, direct_view);
}

} // namespace

// (hookshot) RX_VK_FRAMELOG=<file>: every frame shown, the game's and the made ones, as they go to the screen, 320x180, while
// <file>.go exists (scripts/framelog.py finds the flashes). Taken at the last step and written on a thread of its own, so
// that the game keeps its speed: a shot of every frame at 1280x720 held it at 30 fps, where the flashes may not show.
// A record: u64 time (ns), u32 sequence, u32 kind (0 the game's, 1 made), 320x180 RGBA.
void exposure_log_dump(const char* why);   // (vk_draw.cpp)
void probe_dump_all(const char* why);
bool probe_serial();
namespace {
constexpr uint32_t kLogW = 320, kLogH = 180;
const char* framelog_path() {
    static const char* p = getenv("RX_VK_FRAMELOG");
    return p;
}
struct LogSlot {
    VkBuffer buf = VK_NULL_HANDLE;
    VmaAllocation alloc = nullptr;
    void* map = nullptr;
    bool used = false;
    int64_t at = 0;
    uint32_t seq = 0, kind = 0;
};
LogSlot log_slots[3];
VkImage log_image = VK_NULL_HANDLE;
VmaAllocation log_image_alloc = nullptr;
std::mutex log_mu;
std::condition_variable log_cv;
std::deque<std::vector<uint8_t>> log_queue;
std::atomic<int> log_state{0};   // 0: not tried; 1: complete sink; -1: disabled after a failure
void framelog_resources_drop() {
    for (LogSlot& l : log_slots) {
        if (l.buf) destroy_buffer(l.buf, l.alloc);
        l = LogSlot{};
    }
    if (log_image) destroy_image(log_image, log_image_alloc);
    log_image = VK_NULL_HANDLE;
    log_image_alloc = nullptr;
}
void framelog_write(FILE* f) {
    SetThreadDescription(GetCurrentThread(), L"mkw_vk frame log");
    for (;;) {
        std::vector<uint8_t> rec;
        {
            std::unique_lock lk(log_mu);
            log_cv.wait(lk, [] { return !log_queue.empty(); });
            rec = std::move(log_queue.front());
            log_queue.pop_front();
        }
        const size_t written = fwrite(rec.data(), 1, rec.size(), f);
        if (written != rec.size() || fflush(f) != 0) {
            {
                std::lock_guard lk(log_mu);
                log_state.store(-1, std::memory_order_release);
                log_queue.clear();   // (no worker remains to consume these; don't retain the queued CPU pictures)
            }
            logf("vk: frame log stopped: %s failed (wrote %zu of %zu bytes)", written == rec.size() ? "flush" : "write",
                 written, rec.size());
            fclose(f);
            return;
        }
    }
}
void framelog_enqueue(std::vector<uint8_t> rec) {
    std::lock_guard lk(log_mu);
    if (log_state.load(std::memory_order_acquire) > 0 && log_queue.size() < 512) {
        log_queue.push_back(std::move(rec));
        log_cv.notify_one();
    }
}
bool framelog_ready() {
    const int state = log_state.load(std::memory_order_acquire);
    if (state) return state > 0;
    log_state.store(-1, std::memory_order_release);
    if (!framelog_path()) return false;
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = VK_FORMAT_R8G8B8A8_UNORM;
    ci.extent = {kLogW, kLogH, 1};
    ci.mipLevels = ci.arrayLayers = 1;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    if (!create_image(ci, log_image, log_image_alloc)) {
        framelog_resources_drop();
        logf("vk: frame log disabled: the readback image could not be made");
        return false;
    }
    for (LogSlot& l : log_slots) {
        if (!create_buffer_cached((uint64_t)kLogW * kLogH * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, l.buf, l.alloc, &l.map) || !l.map) {
            framelog_resources_drop();
            logf("vk: frame log disabled: the complete mapped readback set could not be made");
            return false;
        }
    }
    FILE* f = fopen(framelog_path(), "wb");
    if (!f) {
        framelog_resources_drop();
        logf("vk: frame log disabled: cannot open %s", framelog_path());
        return false;
    }
    // Other presenters are serialized by present_image_mu. Publish before the worker can report a failure.
    log_state.store(1, std::memory_order_release);
    try {
        std::thread([f] { framelog_write(f); }).detach();
    } catch (const std::exception& e) {
        log_state.store(-1, std::memory_order_release);
        fclose(f);
        framelog_resources_drop();
        logf("vk: frame log disabled: writer thread could not start (%s)", e.what());
        return false;
    }
    logf("vk: the frame log writes every frame shown to %s", framelog_path());
    return true;
}
bool framelog_on() {
    static bool on = false;
    static int64_t next = 0;
    if (!framelog_path()) return false;
    const int64_t now = prof_now();
    if (now >= next) {
        next = now + 250000000;
        const std::string go = std::string(framelog_path()) + ".go";
        on = GetFileAttributesA(go.c_str()) != INVALID_FILE_ATTRIBUTES;
    }
    return on && framelog_ready();
}
void framelog_collect(int slot) {
    LogSlot& l = log_slots[slot];
    if (!l.used) return;
    l.used = false;
    // present_image waited for this slot's fence; expose its GPU writes before the CPU copy.
    invalidate_mapped(l.alloc, (uint64_t)kLogW * kLogH * 4);
    std::vector<uint8_t> rec(16 + (size_t)kLogW * kLogH * 4);
    std::memcpy(rec.data(), &l.at, 8);
    std::memcpy(rec.data() + 8, &l.seq, 4);
    std::memcpy(rec.data() + 12, &l.kind, 4);
    std::memcpy(rec.data() + 16, l.map, (size_t)kLogW * kLogH * 4);
    {   // (a picture blown out to white: what the exposure was)
        const uint8_t* q = rec.data() + 16;
        uint64_t sum = 0;
        const size_t n = (size_t)kLogW * kLogH;
        for (size_t i = 0; i < n; i += 7) sum += q[i * 4 + 1];
        const uint64_t mean = sum / (n / 7 + 1);
        static uint64_t last_mean;
        if (mean > 225 && last_mean < 215 && l.kind == 0) {
            logf("vk: white frame: frame log sequence %u, present %lld, brightness %llu after %llu", l.seq, (long long)presents,
                 (unsigned long long)mean, (unsigned long long)last_mean);
            char line[700];
            int n2 = snprintf(line, sizeof line, "vk: white detail: textures made by present (now %lld):", (long long)made_ring_present);
            for (int k = 30; k >= 0 && n2 < 660; k--) {
                const int64_t at = made_ring_present - k;
                const uint64_t a = made_ring[(at - 1) & 31], b = made_ring[at & 31];
                if (b > a) n2 += snprintf(line + n2, sizeof line - n2, " -%d:%llu", k, (unsigned long long)(b - a));
            }
            logf("%s", line);
            exposure_log_dump("white frame");
            probe_dump_all("white frame");
        }
        if (l.kind == 0 && mean * 2 < last_mean && last_mean > 40) {   // (a picture gone dark for a frame, the HUD on it)
            static int dark_left = 6;
            if (dark_left > 0) {
                dark_left--;
                logf("vk: dark frame: frame log sequence %u, present %lld, brightness %llu after %llu", l.seq, (long long)presents,
                     (unsigned long long)mean, (unsigned long long)last_mean);
                char name[32];
                snprintf(name, sizeof name, "cond_%d.txt", dark_left);
                if (FILE* o = fopen(name, "w")) {
                    for (uint32_t i = 0; i < (1 << 15); i++) {
                        const CondNote& c = cond_ring[(cond_ring_at + i) % (1 << 15)];
                        if (c.addr)
                            fprintf(o, "%lld %#llx %d %llu %llu %d%c", (long long)c.present, (unsigned long long)c.addr, c.mode,
                                    (unsigned long long)c.x, (unsigned long long)c.y, c.on ? 1 : 0, 10);
                    }
                    fclose(o);
                }
                exposure_log_dump("dark frame");
                probe_dump_all("dark frame");
                pass_traces_log(texture_frame_no() - 7, texture_frame_no());
            }
        }
        if (l.kind == 0) last_mean = mean;
    }
    static const bool drop = getenv("RX_VK_FRAMELOG_DROP") != nullptr;
    if (drop) return;
    framelog_enqueue(std::move(rec));
}
} // namespace
std::atomic<uint32_t> present_kind{0};   // (vk_framegen.cpp: 1 while a made frame goes out)

// (hookshot) One thread in here at a time; the queue's own lock (dev.present_mu, the replay's submits take it too) is held only
// for the submit and the present. Held over vkAcquireNextImageKHR and the fence, it stopped the replay's submits for as long as
// the swapchain had no image free, twice a frame with frame interpolation.
std::mutex present_image_mu;
static void present_out(uint32_t index, std::unique_lock<std::mutex>& qk);   // (present_image's end, below)
thread_local PendingPresent* present_defer_into = nullptr;
void present_image(VkImage src, uint32_t sw, uint32_t sh, const int crop[4], VkImageView view) {
    std::lock_guard lk(present_image_mu);
    vr::probe_tick();
    if (src) vr::present(src, sw, sh, crop);
    static int hdr_was = -1;   // (the Insert menu's HDR as the swapchain was made for it; -1: not yet looked at)
    const bool hdr_changed = hdr_was >= 0 && hdr_was != opts.hdr;   // (turned on or off: the swapchain's format)
    if (hdr_was < 0) hdr_was = opts.hdr;
    const uint64_t want = dev.swap_resize.exchange(0, std::memory_order_acquire);
    if (want || !dev.swapchain || reflex_swapchain_stale() || hdr_changed) {   // (resized, gone, Reflex or HDR changed)
        std::lock_guard qk(dev.present_mu);
        const uint32_t width = want ? (uint32_t)(want >> 32) : dev.swap_extent.width;
        const uint32_t height = want ? (uint32_t)want : dev.swap_extent.height;
        if (!create_swapchain(width, height)) {
            // Retry without replacing a newer resize published while recreation was in progress.
            uint64_t empty = 0;
            dev.swap_resize.compare_exchange_strong(empty, ((uint64_t)width << 32) | height,
                                                   std::memory_order_release, std::memory_order_relaxed);
            return;   // (the old swapchain may already be retired; do not acquire or reset a frame fence)
        }
        hdr_was = opts.hdr;
    }
    if (src) {   // (the Window scaling filter's row: whether the window is the picture's size. Only the game's frames and the
                 // made ones: the start-up screen (no src) is drawn at the window's own size, which made the row read "Not
                 // needed" while the game's output was another size; until the first frame the size is not known: needed)
        const bool cropped = crop && crop[2] > 0 && crop[3] > 0;
        menu_note_present_sizes(cropped ? (uint32_t)crop[2] : sw, cropped ? (uint32_t)crop[3] : sh, dev.swap_extent.width,
                                dev.swap_extent.height);
    }
    menu_tick();
    Frame& f = frames[frame_index];
    const int log_slot = frame_index;
    frame_index = (frame_index + 1) % 3;
    if (vkWaitForFences(dev.device, 1, &f.done, VK_TRUE, UINT64_MAX) == VK_ERROR_DEVICE_LOST) device_lost();
    framelog_collect(log_slot);
    uint32_t index = 0;
#ifdef _WIN32
    const uint64_t acquire_timeout = UINT64_MAX;
#else
    // (linux: a Wayland compositor gives no image back while the window cannot be seen: its screen asleep, another workspace. A wait
    // without end there held the presentation thread, and with it the game, until the screen came back. 100 ms, then the window is
    // taken as not shown: frames are not waited for and not shown, and the game goes on at its speed, as on Windows, until an image
    // comes back)
    static bool unseen = false;
    const uint64_t acquire_timeout = unseen ? 0 : 100000000ull;
#endif
    VkResult res = vkAcquireNextImageKHR(dev.device, dev.swapchain, acquire_timeout, f.acquired, VK_NULL_HANDLE, &index);
    if (res == VK_ERROR_DEVICE_LOST) device_lost();
#ifndef _WIN32
    if (res == VK_TIMEOUT || res == VK_NOT_READY) {
        if (!unseen) logf("vk: the window is not shown (no image back in 100 ms): frames are not shown until it is");
        unseen = true;
        return;
    }
    if (unseen && (res == VK_SUCCESS || res == VK_SUBOPTIMAL_KHR)) {
        unseen = false;
        logf("vk: the window is shown again");
    }
#endif
    if (res == VK_ERROR_OUT_OF_DATE_KHR) {
        std::lock_guard qk(dev.present_mu);
        create_swapchain(dev.swap_extent.width, dev.swap_extent.height);
        return;
    }
    if (res != VK_SUCCESS && res != VK_SUBOPTIMAL_KHR) return;
    if (vkResetFences(dev.device, 1, &f.done) == VK_ERROR_DEVICE_LOST) device_lost();
    if (vkResetCommandPool(dev.device, f.pool, 0) == VK_ERROR_DEVICE_LOST) device_lost();
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(f.cmd, &bi) == VK_ERROR_DEVICE_LOST) device_lost();
    VkImage dst = dev.swap_images[index];
    full_barrier(f.cmd);   // (after the game's work for the frame: earlier submits on this queue)
    image_barrier(f.cmd, dst, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    const PresentAspect aspect{src && host && host->aspect_ratio ? host->aspect_ratio() : 0.0f};
    const PresentRect rect = present_rect(sw, sh, crop, dev.swap_extent.width, dev.swap_extent.height, aspect);
    [[maybe_unused]] uint32_t trace_route = 0; // direct; 1 scaled, 2 clear, 3 HDR conversion
    const bool has_bars = rect.x || rect.y || rect.w != dev.swap_extent.width || rect.h != dev.swap_extent.height;
    if (has_bars) {
        const VkClearColorValue black{0.0f, 0.0f, 0.0f, 1.0f};
        const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(f.cmd, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &black, 1, &range);
        // The following blit overlaps the clear in the content rectangle.
        image_barrier(f.cmd, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    }
    const VkImage scaled = src && view ? framegen_scale(f.cmd, view, src, sw, sh, crop, rect.w, rect.h) : VK_NULL_HANDLE;
    // (hookshot) the Insert menu's HDR: the picture (scaled, or the game's own with its crop) to HDR10, copied in as the SDR one is
    const VkImage hdr_image = dev.hdr && src && view && !present_native_pq.load(std::memory_order_relaxed)
        ? (scaled ? framegen_hdr(f.cmd, framegen_scaled_view(), rect.w, rect.h, nullptr, rect.w, rect.h)
                  : framegen_hdr(f.cmd, view, sw, sh, crop, rect.w, rect.h))
        : VK_NULL_HANDLE;
    if (hdr_image) {
        trace_route = 3;
        full_barrier(f.cmd);
        blit_crop_to_rect(f.cmd, hdr_image, rect.w, rect.h, nullptr, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, rect);
    } else if (scaled) {
        trace_route = 1;
        blit_crop_to_rect(f.cmd, scaled, rect.w, rect.h, nullptr, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, rect);
    } else if (src) {
        blit_crop_to_rect(f.cmd, src, sw, sh, crop, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, rect);
    } else {
        trace_route = 2;
        const VkClearColorValue black{};
        const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(f.cmd, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &black, 1, &range);
    }
    if (src && framelog_on()) {
        static uint32_t seq;
        LogSlot& l = log_slots[log_slot];
        image_barrier(f.cmd, log_image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        blit_crop(f.cmd, src, sw, sh, crop, log_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, kLogW, kLogH);
        image_barrier(f.cmd, log_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        const VkBufferImageCopy c{0, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {kLogW, kLogH, 1}};
        vkCmdCopyImageToBuffer(f.cmd, log_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, l.buf, 1, &c);
        capture_host_read_barrier(f.cmd, l.buf, (VkDeviceSize)kLogW * kLogH * 4);
        l.used = true, l.at = host->now_ns(), l.seq = seq++, l.kind = present_kind.load();
    }
    // Geometry trace: after choosing the actual presentation route, without frame-log pixel readback.
    static const bool trace_geometry = getenv("RX_VK_PRESENT_LOG") && atoi(getenv("RX_VK_PRESENT_LOG")) != 0;
    if (trace_geometry) {
        static PresentGeometryTrace trace;
        PresentGeometrySample sample;
        const PresentContentSize content = present_content_size(sw, sh, crop);
        sample.source = {sw, sh};
        sample.content = {content.w, content.h};
        sample.output = {dev.swap_extent.width, dev.swap_extent.height};
        sample.destination = {rect.x, rect.y, rect.w, rect.h};
        sample.crop_supplied = crop != nullptr;
        if (crop) std::copy_n(crop, 4, sample.crop.begin());
        sample.sampled_view = view != VK_NULL_HANDLE;
        sample.host_aspect(src && host && host->aspect_ratio, aspect.ratio);
        sample.kind = src ? std::min(present_kind.load(), 1u) : 2;
        sample.route = trace_route;
        sample.filter = opts.upscale;
        sample.sharpness = opts.sharpness;
        sample.framegen = opts.framegen;
        const PresentTraceDecision decision = trace.note(sample);
        if (decision == PresentTraceDecision::Emit) {
            const char* kinds[] = {"game", "made", "startup"};
            const char* routes[] = {"direct", "scaled", "clear", "hdr"};
            const char* aspects[] = {"unavailable", "invalid", "valid"};
            logf("vk: present geometry: kind=%s source=%ux%u crop=%s:%d,%d %dx%d content=%ux%u output=%ux%u "
                 "destination=%u,%u %ux%u host-aspect=%s:%.9g route=%s view=%d filter-request=%d sharpness-request=%d framegen-request=%d",
                 kinds[sample.kind], sw, sh, crop ? "supplied" : "none", sample.crop[0], sample.crop[1], sample.crop[2], sample.crop[3],
                 content.w, content.h, sample.output[0], sample.output[1], rect.x, rect.y, rect.w, rect.h,
                 aspects[sample.aspect_state], sample.aspect, routes[sample.route], int(sample.sampled_view), sample.filter,
                 sample.sharpness, sample.framegen);
        } else if (decision == PresentTraceDecision::Limit) {
            logf("vk: present geometry: limit of %u changes reached; further geometry messages suppressed", PresentGeometryTrace::kLimit);
        }
    }
    menu_record(f.cmd, index, dev.swap_extent.width, dev.swap_extent.height);   // (hookshot) Dear ImGui's overlays, over all of it
    image_barrier(f.cmd, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
    if (vkEndCommandBuffer(f.cmd) == VK_ERROR_DEVICE_LOST) device_lost();
    // after the image's acquisition. (Not on the timeline: one queue runs submits in order, and a wait for a value
    // another thread had counted but not yet submitted blocked the queue for good)
    VkSemaphoreSubmitInfo waits[1] = {{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO}};
    waits[0].semaphore = f.acquired;
    waits[0].stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    VkSemaphoreSubmitInfo signal{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    signal.semaphore = dev.swap_rendered[index];
    signal.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    VkCommandBufferSubmitInfo cb{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    cb.commandBuffer = f.cmd;
    VkSubmitInfo2 si{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    si.waitSemaphoreInfoCount = 1;
    si.pWaitSemaphoreInfos = waits;
    si.commandBufferInfoCount = 1;
    si.pCommandBufferInfos = &cb;
    si.signalSemaphoreInfoCount = 1;
    si.pSignalSemaphoreInfos = &signal;
    std::unique_lock qk(dev.present_mu);
    if (!pend.cbs.empty()) flush_reason = FR_PRESENT;
    queue_flush_locked();
    static int64_t presents_counted, st_flushes0, st_cbs0, r0[FR_COUNT];
    if (st_flushes && ++presents_counted % 120 == 0) {
        logf("vk: batched submits: %lld flushes, %.1f command buffers each, over 120 presents; by reason: poll %lld, wait %lld, "
             "slot %lld, present %lld, cap %lld, value %lld, other %lld", (long long)(st_flushes - st_flushes0),
             (double)(st_batched_cbs - st_cbs0) / std::max<int64_t>(1, st_flushes - st_flushes0),
             (long long)(st_reason[FR_POLL] - r0[FR_POLL]), (long long)(st_reason[FR_WAIT] - r0[FR_WAIT]),
             (long long)(st_reason[FR_SLOT] - r0[FR_SLOT]), (long long)(st_reason[FR_PRESENT] - r0[FR_PRESENT]),
             (long long)(st_reason[FR_CAP] - r0[FR_CAP]), (long long)(st_reason[FR_VALUE] - r0[FR_VALUE]),
             (long long)(st_reason[FR_OTHER] - r0[FR_OTHER]));
        st_flushes0 = st_flushes, st_cbs0 = st_batched_cbs;
        std::memcpy(r0, st_reason, sizeof r0);
    }
    reflex_submit_ends();
    res = vkQueueSubmit2(dev.queue, 1, &si, f.done);
    if (res == VK_ERROR_DEVICE_LOST) device_lost();
    if (present_defer_into) {   // (present ahead: the copy is queued; present_commit shows it at its time)
        present_defer_into->used = true;
        present_defer_into->index = index;
        present_defer_into->done = f.done;
        present_defer_into->swapchain = dev.swapchain;
        return;
    }
    present_out(index, qk);
}

bool present_commit(const PendingPresent& p) {
    if (!p.used) return false;
    std::lock_guard lk(present_image_mu);
    std::unique_lock qk(dev.present_mu);
    if (p.swapchain != dev.swapchain) return false;   // (made again since the copy: that image is gone)
    present_out(p.index, qk);
    return true;
}

// The present of a swapchain image whose copy is submitted, as present_image ended before; qk holds dev.present_mu
static void present_out(uint32_t index, std::unique_lock<std::mutex>& qk) {
    VkResult res;
    VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    pi.pNext = reflex_present_id(nullptr);   // (NVIDIA Reflex: the frame's present id, and its present's markers)
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores = &dev.swap_rendered[index];
    pi.swapchainCount = 1;
    pi.pSwapchains = &dev.swapchain;
    pi.pImageIndices = &index;
    antilag_present();   // (AMD Anti-Lag 2: the frame's present stage)
    // (hookshot) vkQueuePresentKHR blocked a quarter of the display thread's time with frame interpolation at 4x (a 14700K and an
    // RTX 5070 Ti), and with the queue's lock held there the game's main thread waited ~9% of its time and the presentation thread
    // ~15% to submit. With a queue of their own (dev.present_queue, a compute family's: vk_main.cpp create_device) the presents go
    // out with the submits' lock let go: the blit above signals swap_rendered on the replay's queue (submitted already, as a binary
    // semaphore's wait needs). (Waiting for the blit's fence before the present, with the lock let go, was worse.)
    if (dev.present_queue) {
        qk.unlock();
        std::lock_guard pq(dev.present_queue_mu);
        res = vkQueuePresentKHR(dev.present_queue, &pi);
    } else {
        res = vkQueuePresentKHR(dev.queue, &pi);
    }
    if (res == VK_ERROR_DEVICE_LOST) device_lost();
    reflex_presented();
    if (res < 0 && res != VK_ERROR_OUT_OF_DATE_KHR) {   // (a present that failed: said, a few times)
        static int logged;
        if (logged++ < 8) logf("vk: present failed (VkResult %d%s)", (int)res, dev.present_queue ? ", on the presents' own queue" : "");
    }
    {   // (RX_VK_FRAME_LOG=<file>: one CSV line a present: frame, seconds since the first, frame time in ms. Frame
        // pacing on the Steam Deck, where MangoHud's log stays empty under gamescope)
        static FILE* fl = getenv("RX_VK_FRAME_LOG") ? std::fopen(getenv("RX_VK_FRAME_LOG"), "w") : nullptr;
        static std::chrono::steady_clock::time_point first, prev;
        static uint64_t frames;
        if (fl) {
            const auto t = std::chrono::steady_clock::now();
            if (!frames) first = prev = t, std::fputs("frame,time_s,frametime_ms\n", fl);
            std::fprintf(fl, "%llu,%.4f,%.3f\n", (unsigned long long)frames, std::chrono::duration<double>(t - first).count(),
                         std::chrono::duration<double, std::milli>(t - prev).count());
            prev = t;
            if (++frames % 60 == 0) std::fflush(fl);
        }
    }
    if (res == VK_ERROR_OUT_OF_DATE_KHR || res == VK_SUBOPTIMAL_KHR) {
        if (!qk.owns_lock()) qk.lock();
        create_swapchain(dev.swap_extent.width, dev.swap_extent.height);
    }
}


namespace {

void set_texture_pool_size(int) {}
void read_texels(const nvn_tex* t, unsigned level, unsigned layer0, const NVNcopyRegion* r, uint8_t* dst, uint64_t row,
                 uint64_t img) {
    finish();   // (what the submits before it wrote)
    std::lock_guard lk(replay_mu);
    {   // (RX_VK_LOG_READ_TEXELS=1: the CPU reading textures back, 200 lines: are they ones the GPU wrote?)
        static const bool log_reads = getenv("RX_VK_LOG_READ_TEXELS") != nullptr;
        static int logged;
        if (log_reads && t && logged < 200) {
            logged++;
            Texture* tex = texture_get(t, false);
            logf("vk: ReadTexels %#llx %ux%ux%u NVN %#x level %u layer %u region (%d,%d,%d %dx%dx%d) row %llu: %s", (unsigned long long)t->gpu,
                 t->d.width, t->d.height, t->d.depth, t->d.format, level, layer0, r ? r->xoffset : 0, r ? r->yoffset : 0,
                 r ? r->zoffset : 0, r ? r->width : 0, r ? r->height : 0, r ? r->depth : 0, (unsigned long long)row,
                 !tex ? "no image" : tex->gpu_written ? (tex->released ? "rendered, finalized" : "RENDERED (guest memory is stale)")
                                                     : "uploaded");
        }
    }
    texture_read_texels(t, level, layer0, r, dst, row, img, false);
}
/// (host: virtual pages [gpu, gpu + size) are about to be unmapped or remapped) what the GPU wrote into textures there
/// goes into their memory first, as on the console, where it is there all along (vk_texture.cpp, texture_flush_to_guest)
bool replay_behind();
void vmap_changing(uint64_t gpu, uint64_t size) {
    if (!size) return;
    static const bool log_calls = getenv("RX_VK_LOG_VMAP_CHANGING") != nullptr;   // (each call, 300 lines)
    static std::atomic<int> logged{0};
    // (the list of GPU-written textures is complete only once the replay has caught up: MK World unmaps a tile's pool
    // a frame or two after it submitted the drawing, which the replay may not have recorded yet)
    const bool listed = texture_gpu_written_in(gpu, size);
    bool behind = false, listed_after = listed;
    if (!listed) {
        behind = replay_behind();
        if (behind) {
            wait_queue_empty();
            listed_after = texture_gpu_written_in(gpu, size);
        }
    }
    if (log_calls && logged++ < 300)
        logf("vk: vmap changing %#llx..%#llx: listed %d, replay behind %d, listed after %d", (unsigned long long)gpu,
             (unsigned long long)(gpu + size), (int)listed, (int)behind, (int)listed_after);
    if (!listed_after) return;
    wait_queue_empty();   // (the work the game recorded before: submitted)
    std::lock_guard lk(replay_mu);
    const int n = texture_flush_to_guest(gpu, size);
    if (log_calls && logged < 300) logf("vk: vmap changing %#llx: %d written back", (unsigned long long)gpu, n);
}
void write_texels(const nvn_tex* t, unsigned level, unsigned layer0, const NVNcopyRegion* r, const uint8_t* src,
                  uint64_t row, uint64_t img) {
    finish();   // (after the submits before it read the texture)
    std::lock_guard lk(replay_mu);
    texture_read_texels(t, level, layer0, r, const_cast<uint8_t*>(src), row, img, true);
}
void stats(int64_t* draws, int64_t* skipped, int64_t* pipelines) {
    int64_t v[4] = {0, 0, 0, 0};
    draw_stats(&v[0], &v[1], &v[2], &v[3]);
    if (draws) *draws = v[0];
    if (skipped) *skipped = v[1];
    if (pipelines) *pipelines = v[3];
}
void register_texture_locked(int id, const nvn_tex* t, const nvn_view* v) { register_texture(id, t, v); }
void do_program_released(const nvn_prog* p) {
    std::lock_guard lk(replay_mu);   // (the replay may be translating it)
    program_released(p);
}

// ---- the replay thread (RX_VK_ASYNC=0: none) ----
// The game's submits and presents are queued and replayed by a thread of the back end's own, so the game's thread
// (MKW's Presentation Thread, 70% busy replaying before) goes on with its frame while the last one is recorded. As on
// the console, where the GPU reads the commands and the memory they name after the submit: the host resolves the
// deferred bindings at the submit and keeps the recordings until rx_nvn_replayed() passes them (nvn_native.c), and
// a fence value is handed out at the submit, signalled when the replay's submission completes. At most two presents
// wait: the game is held at that when the replay falls behind. Finalized textures and programs are queued too, in
// order with the work that may still use them.
enum JobKind { JOB_SUBMIT, JOB_PRESENT, JOB_TEXTURE_RELEASED, JOB_PROGRAM_RELEASED, JOB_REGISTER_TEXTURE, JOB_REGISTER_SAMPLER,
               JOB_EVENT_SIGNAL };
struct Job {
    JobKind kind;
    std::vector<uint64_t> handles;
    nvn_tex tex;
    bool has_tex = false;
    int id = 0;                 // (a registration's slot, its view and sampler)
    nvn_view view{};
    bool has_view = false;
    nvn_samp_par samp{};
    uint64_t ev_gpu = 0;        // (a CPU signal: the event and its value)
    uint32_t ev_value = 0;
    uint64_t value = 0;         // (a submit's or present's timeline value: its fence, handed out as it came)
    uint64_t queue = 0;         // (a submit's queue)
    int crop[4] = {0, 0, 0, 0};
    bool has_crop = false;
    nvn_prog prog;
    std::chrono::steady_clock::time_point queued;
};
std::mutex q_mu;
std::condition_variable q_cv, q_space, q_idle;
std::deque<Job> q_jobs;
int q_presents = 0;
bool q_busy = false;
int presents_ahead() {   // (RX_VK_PRESENTS_AHEAD=<n>: how many presents the game may hand over ahead of the replay)
    static const int n = std::clamp(getenv("RX_VK_PRESENTS_AHEAD") ? atoi(getenv("RX_VK_PRESENTS_AHEAD")) : 2, 1, 8);
    return n;
}

bool async_on() {
    // (hookshot: off by default. OoT rewrites double-buffered data right after a submit with no fence, which lands
    // inside a replay running behind (TotK Recomp's reason to replay on the submitting thread): whole-screen white or
    // black frames, 28 per 1000 gameplay frames, against 1.3 synchronous, at the same frame rate. RX_VK_ASYNC=1: on)
    static const bool on = getenv("RX_VK_ASYNC") && atoi(getenv("RX_VK_ASYNC")) != 0;
    return on;
}

void replay_thread_main() {
    SetThreadDescription(GetCurrentThread(), L"mkw_vk replay");
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
    // (on the core the host keeps free of the game's threads, hos_reserved_cpus: its last physical one)
    if (host->reserved_cpus) {
        if (const uint64_t m = host->reserved_cpus()) {
            SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)m);
            logf("vk: the replay thread runs on logical processors %#llx, kept from the game's threads", (unsigned long long)m);
        }
    }
    for (;;) {
        Job j;
        bool have = false;
        {
            std::unique_lock lk(q_mu);
            if (blocked_n.load())   // (a queue stopped at a wait: the next job, or on without it after a while)
                q_cv.wait_until(lk, blocked_since + std::chrono::milliseconds(hold_ms()),
                                [] { return !q_jobs.empty() || drain_waiters.load() > 0; });
            else
                q_cv.wait(lk, [] { return !q_jobs.empty(); });
            if (!q_jobs.empty()) j = std::move(q_jobs.front()), q_jobs.pop_front(), have = true;
            q_busy = true;
        }
        if (blocked_n.load()) {   // (stopped ones: on now when another thread waits for the replay, or stopped too long)
            if (!have && drain_waiters.load() > 0) drain(2);   // (the jobs before its wait done)
            else if (std::chrono::steady_clock::now() >= blocked_since + std::chrono::milliseconds(hold_ms())) drain(3);
        }
        if (!have) {
            update_blocked();
            std::lock_guard lk(q_mu);
            q_busy = false;
            if (q_jobs.empty() && !blocked_n.load()) q_idle.notify_all();
            continue;
        }
        // (RX_VK_TIMELINE: one frame's jobs every 240 presents: when the game handed each over, how long it waited
        // for the replay thread, how long it took, its draws; the present's time is 0)
        static const bool timeline = getenv("RX_VK_TIMELINE") != nullptr;
        static const bool timeline_late = timeline && atoi(getenv("RX_VK_TIMELINE")) >= 2;
        const auto t_start = std::chrono::steady_clock::now();
        int64_t d0 = 0, x;
        if (timeline) draw_stats(&d0, &x, &x, &x);
        switch (j.kind) {
        case JOB_SUBMIT: submit_job(j.handles, j.value, j.queue); break;
        case JOB_PRESENT:   // (the frame's work whole before it is shown)
            drain(0);
            present_value = j.value;
            do_present(j.has_tex ? &j.tex : nullptr, j.has_crop ? j.crop : nullptr);
            break;
        case JOB_TEXTURE_RELEASED:   // (after the stopped work that may use it)
            drain(1);
            do_texture_released(&j.tex);
            break;
        case JOB_PROGRAM_RELEASED:
            drain(1);
            do_program_released(&j.prog);
            break;
        case JOB_REGISTER_TEXTURE: register_texture_copy(j.id, j.tex, j.has_view ? &j.view : nullptr); break;
        case JOB_REGISTER_SAMPLER: register_sampler_copy(j.id, j.samp); break;
        case JOB_EVENT_SIGNAL:   // (the CPU set an event: the stopped ones may go on)
            ev_values[j.ev_gpu] = j.ev_value;
            burst_begin(0);
            schedule();
            burst_end();
            break;
        }
        update_blocked();
        if (j.kind == JOB_PRESENT && presents % 3600 == 0) log_order_stats();
        if (timeline && (j.kind == JOB_SUBMIT || j.kind == JOB_PRESENT)) {
            struct T { char k; double q, s, e; int64_t d; size_t h; };
            static std::vector<T> frame;
            const auto t_end = std::chrono::steady_clock::now();
            int64_t d1 = 0;
            draw_stats(&d1, &x, &x, &x);
            auto ms = [](auto a) { return std::chrono::duration<double, std::milli>(a.time_since_epoch()).count(); };
            frame.push_back({j.kind == JOB_SUBMIT ? 's' : 'p', ms(j.queued), ms(t_start), ms(t_end), d1 - d0, j.handles.size()});
            if (j.kind == JOB_PRESENT) {
                // (RX_VK_TIMELINE=2: also every frame handed over 20 ms or more after the last one, 200 of them)
                static double last_q = 0;
                static int late_logged = 0;
                const double q_now = frame.back().q;
                const bool late = timeline_late && last_q && q_now - last_q >= 20.0 && late_logged < 200;
                if (late) late_logged++;
                last_q = q_now;
                if (presents % 240 == 0 || late) {
                    const double base = frame.back().q;
                    char line[2048];
                    int n = snprintf(line, sizeof line, "vk: timeline%s (ms from the present's hand-over):",
                                     late ? " of a late frame" : "");
                    for (const T& t : frame)
                        if (n < 1900)
                            n += snprintf(line + n, sizeof line - n, " %c q%+.1f s%+.1f e%+.1f %lldd %zuh;", t.k, t.q - base,
                                          t.s - base, t.e - base, (long long)t.d, t.h);
                    logf("%s", line);
                }
                frame.clear();
            }
        }
        std::lock_guard lk(q_mu);
        if (j.kind == JOB_PRESENT) q_presents--, q_space.notify_all();
        q_busy = false;
        if (q_jobs.empty() && !blocked_n.load()) q_idle.notify_all();
    }
}

void enqueue(Job&& j) {
    j.queued = std::chrono::steady_clock::now();
    std::unique_lock lk(q_mu);
    static bool started;
    if (!started) {
        started = true;
        std::thread(replay_thread_main).detach();
        logf("vk: the replay runs on a thread of its own (RX_VK_ASYNC=0: on the submitting thread)");
    }
    if (j.kind == JOB_PRESENT) {
        q_space.wait(lk, [] { return q_presents < presents_ahead(); });
        q_presents++;
    }
    if (j.kind == JOB_SUBMIT || j.kind == JOB_PRESENT) j.value = ++q_values;
    q_jobs.push_back(std::move(j));
    q_cv.notify_one();
}

void wait_queue_empty() {
    if (!async_on()) {
        // A catch-up (a queue finish, a CPU read back of what the GPU wrote, a pool remapped) used to force every queue
        // stopped at an event wait past it. The work those waits are for is mostly not submitted yet (the waiting thread is the one
        // that submits it), so the stopped submission went on without it and its lit passes read what that work makes before it
        // was made: the scene black, or grey, or blown out for a frame, every other frame for seconds at a time in Kokiri Forest
        // once the game ran at 60 fps (at 4-8 fps the work had always come first). Measured over 70 s of Kokiri gameplay: 10000-
        // 40000 waits gone past and 250-1100 flashing frames before, ~300 and ~65 now, 60 fps either way. A catch-up now lets on the
        // waits that are met; what stays stopped goes on with its own queue's work, a present, or sync_watch's hold_ms as before.
        // RX_VK_CATCHUP_FORCE=1: forced, as before.
        static const bool force = getenv("RX_VK_CATCHUP_FORCE") != nullptr;
        if (force) return sync_drain(2);
        if (!blocked_n.load()) return;
        std::lock_guard lk(sync_order_mu);
        burst_begin(0);
        schedule();
        burst_end();
        update_blocked();
        return;
    }
    std::unique_lock lk(q_mu);
    drain_waiters++;   // (stopped submissions go on now: replay_thread_main)
    q_cv.notify_all();
    q_idle.wait(lk, [] { return q_jobs.empty() && !q_busy && blocked_n.load() == 0; });
    drain_waiters--;
}
/// (work handed to the replay thread and not yet recorded)
bool replay_behind() {
    if (!async_on()) return blocked_n.load() > 0;
    std::lock_guard lk(q_mu);
    return !q_jobs.empty() || q_busy || blocked_n.load() > 0;
}

std::mutex submit_order_mu;   // (captures and submits queued in the same order)
void submit_q(const uint64_t* handles, int n, uint64_t queue) {
    if (!async_on()) {
        std::lock_guard order(sync_order_mu);
        ReplayPin pin;   // (the capture and the replay on the replay's core)
        // (the replay here is on the submitting thread, but the game's other threads go on writing their next frames: a pool
        // synced from guest memory later in the frame read those. The capture is taken here too)
        // (EvilOOT) on by default: without it every mirror sync read guest memory as it was at the replay, the game's later
        // frames. OoT's GPU culling alternates two list sets: every other frame its setup read the next frame's records,
        // counted into the other set, and the foliage, grass, characters and particles of that frame were never drawn.
        // With it the syncs take the submits' captures, as the console's GPU sees its memory. RX_VK_SYNC_CAPTURE=0: off.
        static const bool capture = !getenv("RX_VK_SYNC_CAPTURE") || atoi(getenv("RX_VK_SYNC_CAPTURE")) != 0;
        if (capture) memory_capture_submit();
        // (RX_VK_SUBMIT_DELAY_US=<us> (a test): a console's GPU starts later than this replay does. Does what a submission reads
        // arrive just after it, from the game's other threads?)
        static const int delay_us = getenv("RX_VK_SUBMIT_DELAY_US") ? atoi(getenv("RX_VK_SUBMIT_DELAY_US")) : 0;
        if (delay_us) {
            const auto until = std::chrono::steady_clock::now() + std::chrono::microseconds(delay_us);
            while (std::chrono::steady_clock::now() < until) std::this_thread::yield();
        }
        if (sync_waits_on()) {
            static std::once_flag watch;
            std::call_once(watch, [] { std::thread(sync_watch).detach(); });
            std::vector<uint64_t> h(handles, handles + std::max(n, 0));
            const uint64_t value = ++q_values;
            if (queue) {
                q_last[queue] = value;
                value_queue[value] = queue;
                while (value_queue.size() > 8192) value_queue.erase(value_queue.begin());
            }
            submit_job(h, value, queue);
            update_blocked();
            return;
        }
        return do_submit(handles, n);
    }
    std::lock_guard lk(submit_order_mu);
    memory_capture_submit();   // (the pages as they are now, on the game's thread: vk_memory.cpp)
    Job j{JOB_SUBMIT};
    j.handles.assign(handles, handles + std::max(n, 0));
    j.queue = queue;
    enqueue(std::move(j));
}
void submit(const uint64_t* handles, int n) {
    submit_q(handles, n, 0);
}
void present(const nvn_tex* t, const int crop[4]) {
    if (!async_on()) {
        std::lock_guard order(sync_order_mu);
        ReplayPin pin;   // (the frame's last replays and the present on the replay's core)
        if (sync_waits_on()) {
            drain(0);   // (the frame's work whole before it is shown)
            update_blocked();
            present_value = ++q_values;
            return do_present(t, crop);
        }
        return do_present(t, crop);
    }
    Job j{JOB_PRESENT};
    if (t) j.tex = *t, j.has_tex = true;
    if (crop) std::memcpy(j.crop, crop, sizeof j.crop), j.has_crop = true;
    enqueue(std::move(j));
}
void texture_released_of(const nvn_tex* t) {
    if (!t) return;
    tex_pages_add(t, -1);
    if (!async_on()) {
        // (PR 430: after the stopped work that may use it, when it goes on. The lock is held_mu, only while queues are
        // stopped: sync_order_mu, taken on every release, made the thread finalizing textures (OptimizeChangeBank) wait for whole
        // submissions being replayed under it, 45% of its time in Kokiri. A release just after the stopped work went on waits
        // for the next update_blocked: the next submit or present)
        if (hold_releases() && sync_waits_on() && blocked_n.load()) {
            {
                std::lock_guard hl(held_mu);
                held_tex_releases.push_back(*t);
            }
            if (++st_held_releases % 2000 == 1) logf("vk: releases kept for stopped work: %lld", (long long)st_held_releases.load());
            return;
        }
        sync_drain(1);   // (after the stopped work that may use it)
        return do_texture_released(t);
    }
    Job j{JOB_TEXTURE_RELEASED};
    j.tex = *t, j.has_tex = true;
    enqueue(std::move(j));
}
void program_released_locked(const nvn_prog* p) {
    if (!p) return;
    if (!async_on()) {
        if (hold_releases() && sync_waits_on() && blocked_n.load()) {   // (PR 430: as texture releases, on held_mu)
            {
                std::lock_guard hl(held_mu);
                held_prog_releases.push_back(*p);
            }
            st_held_releases++;
            return;
        }
        sync_drain(1);
        return do_program_released(p);
    }
    Job j{JOB_PROGRAM_RELEASED};
    j.prog = *p;
    enqueue(std::move(j));
}
// ---- texture and sampler registrations: in order with the submits ----
// A registration took effect at once, on the game's thread, while the replay recorded submits up to two presents
// older: the game, which registers the slots of its next frames as it builds them (sets of render targets at each
// dynamic-resolution step), had the frame being recorded sample the next ones' textures, of another size: its
// picture went to pieces below or above a line (edges clamped into stripes, the sky's or fog's colour, the shadows
// dark) for a frame, with the size changes (every few frames when the resolution swung). Now they are jobs, applied
// where the game made them among its submits, as finalized textures already were. RX_VK_REGISTER_AT_ONCE=1: as before.
bool register_at_once() {
    static const bool v = getenv("RX_VK_REGISTER_AT_ONCE") != nullptr;
    return v;
}
void register_texture_queued(int id, const nvn_tex* t, const nvn_view* v) {
    if (!async_on() || register_at_once()) return register_texture_locked(id, t, v);
    if (id < 0 || id >= (1 << 20) || !t) return;
    t = (const nvn_tex*)descriptor_ok(t, sizeof(nvn_tex));
    if (!t) return;
    v = (const nvn_view*)descriptor_ok(v, sizeof(nvn_view));
    Job j{JOB_REGISTER_TEXTURE};
    j.id = id, j.tex = *t;
    if (v) j.view = *v, j.has_view = true;
    enqueue(std::move(j));
}
void event_signal_queued(uint64_t gpu, uint32_t value) {
    if (!async_on()) {
        std::lock_guard lk(ev_cpu_mu);
        ev_cpu_pending.emplace_back(gpu, value);
        ev_cpu_any.store(true, std::memory_order_release);
        return;   // (sync_watch or the next submit lets the stopped ones on: not this thread, ~1.7 ms a frame of the critical path)
    }
    Job j{JOB_EVENT_SIGNAL};
    j.ev_gpu = gpu, j.ev_value = value;
    enqueue(std::move(j));
}
void register_sampler_queued(int id, const nvn_samp_par* p) {
    if (!async_on() || register_at_once()) return register_sampler(id, p);
    if (id < 0 || id >= (1 << 12) || !p) return;
    p = (const nvn_samp_par*)descriptor_ok(p, sizeof(nvn_samp_par));
    if (!p) return;
    Job j{JOB_REGISTER_SAMPLER};
    j.id = id, j.samp = *p;
    enqueue(std::move(j));
}
uint64_t fence() { return async_on() || sync_waits_on() ? q_values.load() : pushed.load(); }
uint64_t get_pushed() { return fence(); }

// (a parallel replay's segments) each recording's init command pool and buffer: made with RX_VK_PARALLEL only (par_threads
// is the process's), last in replay_start, which runs once a process. All or none: a failure gives back what was made and
// the start fails, as with the recordings' other pools. Kept as long as the device, like those
void init_pools_destroy() {
    for (Recording& r : recordings) {
        if (r.init_pool) vkDestroyCommandPool(dev.device, r.init_pool, nullptr);   // (its command buffer with it)
        r.init_pool = VK_NULL_HANDLE, r.init = VK_NULL_HANDLE;
    }
}
bool init_pools_make(const VkCommandPoolCreateInfo& pi) {
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    for (Recording& r : recordings) {
        VkCommandPool pool = VK_NULL_HANDLE;   // (the outputs are kept only on success)
        VkCommandBuffer cb = VK_NULL_HANDLE;
        if (vkCreateCommandPool(dev.device, &pi, nullptr, &pool) != VK_SUCCESS) {
            init_pools_destroy();
            return false;
        }
        r.init_pool = ai.commandPool = pool;
        if (vkAllocateCommandBuffers(dev.device, &ai, &cb) != VK_SUCCESS) {
            init_pools_destroy();
            return false;
        }
        r.init = cb;
    }
    return true;
}

} // namespace

bool replay_start() {
    menu_init();
    if (!memory_init() || !vma_init(instance_proc())) return false;
    VkSemaphoreTypeCreateInfo ti{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    ti.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    VkSemaphoreCreateInfo tsi{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    tsi.pNext = &ti;
    if (vkCreateSemaphore(dev.device, &tsi, nullptr, &timeline) != VK_SUCCESS) return false;
    if (vkCreateSemaphore(dev.device, &tsi, nullptr, &batch_tl) != VK_SUCCESS) batch_tl = VK_NULL_HANDLE;
    if (vkCreateSemaphore(dev.device, &tsi, nullptr, &rec_timeline) != VK_SUCCESS) rec_timeline = VK_NULL_HANDLE;
    VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pi.queueFamilyIndex = dev.queue_family;
    pi.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    int slot = 0;
    for (Recording& r : recordings) {
        r.slot = slot++;
        VkFenceCreateInfo rfi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        if (vkCreateFence(dev.device, &rfi, nullptr, &r.fence) != VK_SUCCESS) return false;
        if (vkCreateCommandPool(dev.device, &pi, nullptr, &r.pool) != VK_SUCCESS) return false;
        if (vkCreateCommandPool(dev.device, &pi, nullptr, &r.upload_pool) != VK_SUCCESS) return false;
        ai.commandPool = r.pool;
        ai.commandBufferCount = 1;
        vkAllocateCommandBuffers(dev.device, &ai, &r.cmd);
        ai.commandPool = r.upload_pool;
        vkAllocateCommandBuffers(dev.device, &ai, &r.upload);
    }
    ai.commandBufferCount = 1;
    for (Frame& f : frames) {
        if (vkCreateCommandPool(dev.device, &pi, nullptr, &f.pool) != VK_SUCCESS) return false;
        ai.commandPool = f.pool;
        vkAllocateCommandBuffers(dev.device, &ai, &f.cmd);
        VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        vkCreateFence(dev.device, &fi, nullptr, &f.done);
        VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        vkCreateSemaphore(dev.device, &si, nullptr, &f.acquired);
    }
    if (par_threads() >= 2 && !init_pools_make(pi)) return false;   // (last: nothing after it fails and leaves them made)
    logf("vk: replay ready (%s; push descriptors %s (%u), extended dynamic state 3 %s, custom border colours %s)",
         dev.offscreen ? "offscreen" : "window", dev.push_descriptor ? "yes" : "no", dev.max_push_descriptors,
         dev.eds3_blend ? "yes" : "no", dev.custom_border_color ? "yes" : "no");
    return true;
}

/// (hookshot) The host's camera jitter (nx2/oot_jitter.py, host/rx_backend.c): the scene's size while a temporal upscaler
/// draws, NVIDIA DLSS or AMD FSR 4 (vk_temporal.cpp) or AMD FSR 3.1 (vk_fsr3_inputs.cpp), whichever it is (one at a time: the
/// Upscaler row); 0, no jitter, otherwise
int jitter_option(int id) {
    const int axis = id - HOS_OPT_DLSS_JITTER_W;
    const uint32_t d = temporal_jitter_size(axis);
    return (int)(d ? d : fsr3_inputs::jitter_size(axis));
}

void fill_replay_table(MkwVkRenderer& t) {
    t.stats = stats;
    t.nvn_submit = submit;
    t.nvn_submit_q = submit_q;
    t.nvn_fence_q = fence_q;
    t.nvn_fence = fence;
    t.nvn_pushed = get_pushed;
    t.nvn_replayed = get_replayed;
    t.nvn_wait = wait;
    t.wait_writebacks_reading = wait_writebacks_reading;
    t.nvn_finish = finish;
    t.nvn_present = present;
    t.nvn_register_texture = register_texture_queued;
    t.nvn_event_signal = event_signal_queued;
    t.option = [](int id) -> int {   // (3: the fixed path's scale in 1/10000, vk_menu.cpp; DLAA: full scale, vk_dlss.cpp)
        if (id == 11) return opts.stick_dead_zone;
        if (id == HOS_OPT_CHEAT_HEARTS) return menu_heart_protection();
        if (id == HOS_OPT_CHEAT_FLIGHT) return menu_player_flight();
        if (id == HOS_OPT_CHEAT_ACTION) return menu_take_cheat_action();
        if (id == HOS_OPT_CHEAT_CAMERA) return menu_free_camera();
        if (id == HOS_OPT_HDR_NATIVE) return dev.hdr && opts.hdr == 2;
        if (id == HOS_OPT_HDR_PEAK) return std::clamp(opts.hdr_peak, 400, 10000);
        if (id == HOS_OPT_HDR_WHITE) return std::clamp(opts.hdr_white, 80, 500);
        if (id == HOS_OPT_DISCORD) return opts.discord != 0;
        if (id == HOS_OPT_VOLUME) return opts.volume;
        if (id == HOS_OPT_MUTE_BACKGROUND) return opts.mute_background;
        return id == 0 ? opts.realtime : id == 1 ? menu_is_open() : id == 2 ? resolution_auto() && !temporal_full_scale() : id == 3 ? (temporal_full_scale() ? 10000 : resolution_scale()) : id == 4 ? opts.pad_background : id == 5 ? opts.text_speed : id == 6 ? opts.mouse_native :
               id == 7 ? opts.mouse_sens : id == 8 ? opts.mouse_invert : id == 9 ? opts.rumble : id == 16 ? opts.invert_cam_x : id == 17 ? opts.invert_cam_y : id == HOS_OPT_GYRO ? opts.gyro : id == HOS_OPT_GYRO_SENS_X ? opts.gyro_sens_x : id == HOS_OPT_GYRO_SENS_Y ? opts.gyro_sens_y : id == HOS_OPT_GYRO_INV_X ? opts.gyro_inv_x : id == HOS_OPT_GYRO_INV_Y ? opts.gyro_inv_y : id == HOS_OPT_GYRO_DEADZONE ? opts.gyro_deadzone :
               id == HOS_OPT_DLSS_JITTER_W || id == HOS_OPT_DLSS_JITTER_H || id == HOS_OPT_DLSS_JITTER_PHASES ? jitter_option(id) : 0;
    };
    t.option_set = menu_option_set;
    t.nvn_register_sampler = register_sampler_queued;
    t.nvn_set_texture_pool_size = set_texture_pool_size;
    t.nvn_texture_released = texture_released_of;
    t.nvn_program_released = program_released_locked;
    t.nvn_read_texels = read_texels;
    t.nvn_write_texels = write_texels;
    t.nvn_invalidate = invalidate;
    t.nvn_pool_created = memory_pool_created;
    t.guest_imported = memory_guest_imported;
    t.nvn_vmap_changing = vmap_changing;
    t.nvn_texture_created = texture_created_of;
}

std::atomic<bool> par_active{false};
std::atomic<int64_t> par_lock_wait_ns{0}, par_locks{0};
std::recursive_mutex& par_mutex() {
    return par_mu;
}
int64_t par_now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

} // namespace mkwvk
