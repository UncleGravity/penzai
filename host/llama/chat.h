#pragma once
// FFI shim header — the one Zig↔llama.cpp C++ bridge. Implemented in chat.cpp.

#include <stdbool.h>

struct llama_model;
struct llama_sampler;

enum {
    PENZAI_CHAT_NO_TEMPLATE = -1,
    PENZAI_CHAT_RENDER_FAILED = -2,
};

#ifdef __cplusplus
extern "C" {
#endif

int penzai_chat_format_user(const struct llama_model *model,
                            const char *content,
                            bool enable_thinking,
                            char *out,
                            int out_len);

// Greedy sampler for Penzai's one-sequence backend-sampling path. It returns
// only the sampled token and bypasses llama.cpp's unused sampling PAD when the
// exact supported graph topology is present.
struct llama_sampler *penzai_sampler_init_terminal_greedy(void);

#ifdef __cplusplus
}
#endif
