#include "chat.h"

#include "common/chat.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <string>
#include <utility>

extern "C" int penzai_chat_format_user(const llama_model *model,
                                       const char *content,
                                       bool enable_thinking,
                                       char *out,
                                       int out_len) {
    if (model == nullptr || content == nullptr) {
        return PENZAI_CHAT_RENDER_FAILED;
    }

    try {
        common_chat_templates_ptr tmpls = common_chat_templates_init(model, "");
        if (!tmpls) {
            return PENZAI_CHAT_RENDER_FAILED;
        }
        if (!common_chat_templates_was_explicit(tmpls.get())) {
            return PENZAI_CHAT_NO_TEMPLATE;
        }

        common_chat_templates_inputs inputs;
        inputs.use_jinja = true;
        inputs.add_generation_prompt = true;
        inputs.enable_thinking = enable_thinking;

        common_chat_msg msg;
        msg.role = "user";
        msg.content = content;
        inputs.messages.push_back(std::move(msg));

        const std::string prompt = common_chat_templates_apply(tmpls.get(), inputs).prompt;
        if (prompt.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
            return PENZAI_CHAT_RENDER_FAILED;
        }

        if (out != nullptr && out_len > 0) {
            const size_t n = std::min(static_cast<size_t>(out_len - 1), prompt.size());
            std::memcpy(out, prompt.data(), n);
            out[n] = '\0';
        }

        return static_cast<int>(prompt.size());
    } catch (...) {
        return PENZAI_CHAT_RENDER_FAILED;
    }
}
