#pragma once
// FFI shim header — the one Zig↔llama.cpp C++ bridge. Renders a chat template for
// a single user turn via llama.cpp's common_chat_* API. Implemented in chat.cpp.

#include <stdbool.h>

struct llama_model;

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

#ifdef __cplusplus
}
#endif
