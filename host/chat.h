#pragma once

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
