#pragma once
// ======================================================
// statediff/log.hpp
//
// Thin wrapper around scissor/log.hpp that adds a
// "statediff::" component prefix to all log entries.
//
// Usage:
//   STATEDIFF_INFO("tree created", "leaves", n_leaves)
//   STATEDIFF_TIMER_START(my_op)
//   STATEDIFF_TIMER_PERF(my_op, "op complete", "bytes", size)
// ======================================================

#include <scissor/log.hpp>
#include <cstdlib>
#include <string>

// ======================================================
// Per-level convenience macros
// Component field is "statediff::" + enclosing function name.
// ======================================================
#define _STATEDIFF_COMP ("statediff::" + std::string(__func__)).c_str()

#define STATEDIFF_DEBUG(msg, ...) \
    SCISSOR_LOG(DEBUG, _STATEDIFF_COMP, _SCISSOR_SRC, msg, ##__VA_ARGS__)
#define STATEDIFF_PERF(msg, ...) \
    SCISSOR_LOG(PERF, _STATEDIFF_COMP, _SCISSOR_SRC, msg, ##__VA_ARGS__)
#define STATEDIFF_INFO(msg, ...) \
    SCISSOR_LOG(INFO, _STATEDIFF_COMP, _SCISSOR_SRC, msg, ##__VA_ARGS__)
#define STATEDIFF_WARN(msg, ...) \
    SCISSOR_LOG(WARN, _STATEDIFF_COMP, _SCISSOR_SRC, msg, ##__VA_ARGS__)
#define STATEDIFF_ERROR(msg, ...) \
    SCISSOR_LOG(ERROR, _STATEDIFF_COMP, _SCISSOR_SRC, msg, ##__VA_ARGS__)
// STATEDIFF_FATAL emits a FATAL log entry then calls std::abort().
#define STATEDIFF_FATAL(msg, ...) \
    do { \
        SCISSOR_LOG(FATAL, _STATEDIFF_COMP, _SCISSOR_SRC, msg, ##__VA_ARGS__); \
        std::abort(); \
    } while (0)

// ======================================================
// Timer helpers
// ======================================================
#define STATEDIFF_TIMER_START(name) \
    auto _statediff_t_##name = ::scissor::log::now()

#define STATEDIFF_TIMER_PERF(name, msg, ...) \
    STATEDIFF_PERF(msg, "elapsed_ms", ::scissor::log::elapsed_ms(_statediff_t_##name), ##__VA_ARGS__)
