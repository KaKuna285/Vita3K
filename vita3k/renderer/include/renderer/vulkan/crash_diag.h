// DEBUG BUILD ONLY: records the last Vulkan draws and dumps them to the log when the
// emulator terminates (for example on vk::DeviceLostError). Not meant for upstream.

#pragma once

#include <util/hash.h>
#include <util/log.h>

#include <vulkan/vulkan.hpp>

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <exception>

namespace renderer::vulkan::crash_diag {

enum CrumbFlags : uint32_t {
    CRUMB_MASKUPDATE = 1 << 0,
    CRUMB_INTERLOCK = 1 << 1,
    CRUMB_MASKED_FETCH = 1 << 2,
    CRUMB_ORDERED_FETCH = 1 << 3,
    CRUMB_FRAGMENT_DISABLED = 1 << 4,
};

struct Crumb {
    uint64_t seq = 0;
    uint64_t frame = 0;
    uint64_t vert_hash = 0;
    uint64_t frag_hash = 0;
    uint32_t color_format = 0;
    uint32_t primitive = 0;
    uint32_t index_count = 0;
    uint32_t instance_count = 0;
    uint32_t flags = 0;
};

constexpr size_t CRUMB_COUNT = 256;
constexpr size_t CRUMB_DUMP_COUNT = 48;

inline std::array<Crumb, CRUMB_COUNT> crumbs{};
inline std::atomic<uint64_t> next_seq{ 1 };
inline vk::Queue checkpoint_queue{};
inline bool use_checkpoints = false;
inline std::terminate_handler previous_terminate = nullptr;

inline uint64_t short_hash(const Sha256Hash &hash) {
    uint64_t result = 0;
    for (int i = 0; i < 8; i++)
        result = (result << 8) | hash[i];
    return result;
}

// returns the sequence number of the crumb, also used as the checkpoint marker
inline uint64_t record(const Crumb &crumb) {
    const uint64_t seq = next_seq.fetch_add(1, std::memory_order_relaxed);
    Crumb &slot = crumbs[seq % CRUMB_COUNT];
    slot = crumb;
    slot.seq = seq;
    return seq;
}

inline void log_crumb(const Crumb &c) {
    LOG_CRITICAL("  #{} frame={} prim={} idx={} inst={} color=0x{:08X} vert={:016x} frag={:016x}{}{}{}{}{}",
        c.seq, c.frame, c.primitive, c.index_count, c.instance_count, c.color_format, c.vert_hash, c.frag_hash,
        (c.flags & CRUMB_MASKUPDATE) ? " maskupdate" : "",
        (c.flags & CRUMB_INTERLOCK) ? " interlock" : "",
        (c.flags & CRUMB_MASKED_FETCH) ? " masked_fetch" : "",
        (c.flags & CRUMB_ORDERED_FETCH) ? " ordered_fetch" : "",
        (c.flags & CRUMB_FRAGMENT_DISABLED) ? " frag_disabled" : "");
}

inline void dump() {
    const uint64_t last = next_seq.load() - 1;
    LOG_CRITICAL("[crash_diag] last {} Vulkan draws recorded on the CPU side (newest last):", CRUMB_DUMP_COUNT);
    const uint64_t first = (last > CRUMB_DUMP_COUNT) ? last - CRUMB_DUMP_COUNT + 1 : 1;
    for (uint64_t seq = first; seq <= last; seq++) {
        const Crumb &c = crumbs[seq % CRUMB_COUNT];
        if (c.seq == seq)
            log_crumb(c);
    }

    if (!use_checkpoints) {
        LOG_CRITICAL("[crash_diag] VK_NV_device_diagnostic_checkpoints not available, no GPU-side position");
        return;
    }

    try {
        const auto checkpoints = checkpoint_queue.getCheckpointDataNV();
        LOG_CRITICAL("[crash_diag] GPU checkpoints ({}): the draw the GPU was working on when it failed", checkpoints.size());
        for (const auto &cp : checkpoints) {
            const uint64_t seq = reinterpret_cast<uint64_t>(cp.pCheckpointMarker);
            LOG_CRITICAL("  stage={} marker=#{}", vk::to_string(cp.stage), seq);
            const Crumb &c = crumbs[seq % CRUMB_COUNT];
            if (c.seq == seq)
                log_crumb(c);
        }
    } catch (const std::exception &e) {
        LOG_CRITICAL("[crash_diag] could not read GPU checkpoints: {}", e.what());
    }
}

inline void terminate_handler() {
    try {
        dump();
    } catch (...) {
    }
    if (previous_terminate)
        previous_terminate();
    std::abort();
}

inline void install(vk::Queue queue, bool checkpoints) {
    checkpoint_queue = queue;
    use_checkpoints = checkpoints;
    previous_terminate = std::set_terminate(terminate_handler);
    LOG_INFO("[crash_diag] enabled, GPU checkpoints: {}", checkpoints ? "yes" : "no");
}

} // namespace renderer::vulkan::crash_diag
