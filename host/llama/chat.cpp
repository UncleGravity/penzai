#include "chat.h"

#include "common/chat.h"
#include "llama.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <limits>
#include <string>
#include <utility>

namespace {

struct terminal_greedy_context {
    llama_sampler *stock;
};

static terminal_greedy_context *sampler_context(llama_sampler *smpl) {
    return static_cast<terminal_greedy_context *>(smpl->ctx);
}

static bool graph_contains(ggml_cgraph *gf, const ggml_tensor *tensor) {
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        if (ggml_graph_node(gf, i) == tensor) {
            return true;
        }
    }
    return false;
}

static bool is_single_f32_row(const ggml_tensor *tensor) {
    return tensor != nullptr &&
           tensor->type == GGML_TYPE_F32 &&
           tensor->ne[0] > 0 && tensor->ne[1] == 1 &&
           tensor->ne[2] == 1 && tensor->ne[3] == 1 &&
           tensor->nb[0] == sizeof(float) &&
           ggml_is_contiguous(tensor);
}

static bool is_sampling_pad(const ggml_tensor *pad, const ggml_tensor *real) {
    if (pad == nullptr || real == nullptr || pad->op != GGML_OP_PAD ||
        pad->src[0] != real || pad->src[1] != nullptr ||
        pad->view_src != nullptr || pad->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(pad)) {
        return false;
    }

    if (pad->ne[0] != real->ne[0] || pad->ne[1] != 2 ||
        pad->ne[2] != 1 || pad->ne[3] != 1) {
        return false;
    }

    // ggml_pad(real, 0, 1, 0, 0): one right-side row and nothing else.
    for (int i = 0; i < 9; ++i) {
        const int32_t expected = i == 3 ? 1 : 0;
        if (pad->op_params[i] != expected) {
            return false;
        }
    }
    return true;
}

static ggml_tensor *direct_logits(ggml_cgraph *gf, const llama_sampler_data *data) {
    ggml_tensor *view = data->logits;
    if (view == nullptr || data->probs != nullptr || data->sampled != nullptr ||
        data->candidates != nullptr || view->op != GGML_OP_VIEW ||
        view->view_offs != 0 || view->src[0] == nullptr ||
        view->view_src != view->src[0] || !is_single_f32_row(view)) {
        return nullptr;
    }

    ggml_tensor *pad = view->src[0];
    ggml_tensor *real = pad->src[0];
    if (!is_single_f32_row(real) || !is_sampling_pad(pad, real) ||
        view->ne[0] != real->ne[0] || graph_contains(gf, view) ||
        graph_contains(gf, pad)) {
        return nullptr;
    }

    return real;
}

static const char *terminal_greedy_name(const llama_sampler *) {
    return "penzai-terminal-greedy";
}

static void terminal_greedy_accept(llama_sampler *smpl, llama_token token) {
    llama_sampler_accept(sampler_context(smpl)->stock, token);
}

static void terminal_greedy_apply(llama_sampler *smpl, llama_token_data_array *cur_p) {
    llama_sampler_apply(sampler_context(smpl)->stock, cur_p);
}

static void terminal_greedy_reset(llama_sampler *smpl) {
    llama_sampler_reset(sampler_context(smpl)->stock);
}

static llama_sampler *terminal_greedy_clone(const llama_sampler *) {
    return penzai_sampler_init_terminal_greedy();
}

static void terminal_greedy_free(llama_sampler *smpl) {
    terminal_greedy_context *ctx = sampler_context(smpl);
    llama_sampler_free(ctx->stock);
    delete ctx;
}

static bool terminal_greedy_backend_init(
    llama_sampler *smpl,
    ggml_backend_buffer_type_t buft) {
    llama_sampler *stock = sampler_context(smpl)->stock;
    return stock->iface->backend_init != nullptr &&
           stock->iface->backend_init(stock, buft);
}

static void terminal_greedy_backend_apply(
    llama_sampler *,
    ggml_context *ctx,
    ggml_cgraph *gf,
    llama_sampler_data *data) {
    ggml_tensor *logits = direct_logits(gf, data);
    if (logits == nullptr) {
        logits = data->logits;
    }

    ggml_tensor *sampled = ggml_argmax(ctx, logits);
    ggml_set_name(sampled, "penzai_greedy_argmax");
    data->sampled = sampled;

    // This sampler is terminal: retaining logits here makes llama.cpp record and
    // download the full vocabulary in addition to the selected token.
    data->logits = nullptr;
}

static llama_sampler_i terminal_greedy_iface = {
    /* .name              = */ terminal_greedy_name,
    /* .accept            = */ terminal_greedy_accept,
    /* .apply             = */ terminal_greedy_apply,
    /* .reset             = */ terminal_greedy_reset,
    /* .clone             = */ terminal_greedy_clone,
    /* .free              = */ terminal_greedy_free,
    /* .backend_init      = */ terminal_greedy_backend_init,
    /* .backend_accept    = */ nullptr,
    /* .backend_apply     = */ terminal_greedy_backend_apply,
    /* .backend_set_input = */ nullptr,
};

} // namespace

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

extern "C" llama_sampler *penzai_sampler_init_terminal_greedy() {
    return llama_sampler_init(
        &terminal_greedy_iface,
        new terminal_greedy_context{
            llama_sampler_init_greedy(),
        });
}
