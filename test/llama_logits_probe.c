#include "ggml-backend.h"
#include "llama.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static int fail(const char * message) {
    fprintf(stderr, "llama logits probe: %s\n", message);
    return 1;
}

int main(int argc, char ** argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: %s MODEL CPU_BACKEND PENZAI_BACKEND EXPECTED_TOKEN\n", argv[0]);
        return 2;
    }

    char * token_end = NULL;
    const long expected_raw = strtol(argv[4], &token_end, 10);
    if (token_end == argv[4] || *token_end != '\0' || expected_raw < 0 || expected_raw > INT32_MAX) {
        return fail("invalid expected token");
    }
    const llama_token expected = (llama_token) expected_raw;

    llama_backend_init();
    if (ggml_backend_reg_by_name("CPU") == NULL && ggml_backend_load(argv[2]) == NULL) {
        llama_backend_free();
        return fail("could not load the CPU backend");
    }
    if (ggml_backend_reg_by_name("penzai") == NULL && ggml_backend_load(argv[3]) == NULL) {
        llama_backend_free();
        return fail("could not load the Penzai backend");
    }

    ggml_backend_dev_t device = ggml_backend_dev_by_name("penzai");
    if (device == NULL) {
        llama_backend_free();
        return fail("Penzai device was not registered");
    }

    ggml_backend_dev_t devices[] = { device, NULL };
    struct llama_model_params model_params = llama_model_default_params();
    model_params.devices = devices;
    model_params.n_gpu_layers = 999;
    model_params.split_mode = LLAMA_SPLIT_MODE_LAYER;

    struct llama_model * model = llama_model_load_from_file(argv[1], model_params);
    if (model == NULL) {
        llama_backend_free();
        return fail("model load failed");
    }

    const struct llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t vocab_count = vocab == NULL ? 0 : llama_vocab_n_tokens(vocab);
    if (vocab_count <= 0 || expected >= vocab_count) {
        llama_model_free(model);
        llama_backend_free();
        return fail("invalid vocabulary or scripted winner");
    }

    struct llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = 64;
    context_params.n_batch = 4;
    context_params.n_ubatch = 4;
    context_params.n_seq_max = 1;
    context_params.n_threads = 1;
    context_params.n_threads_batch = 1;
    context_params.op_offload = false;

    struct llama_context * context = llama_init_from_model(model, context_params);
    if (context == NULL) {
        llama_model_free(model);
        llama_backend_free();
        return fail("context initialization failed");
    }

    llama_token input = 1;
    const struct llama_batch batch = llama_batch_get_one(&input, 1);
    if (llama_decode(context, batch) != 0) {
        llama_free(context);
        llama_model_free(model);
        llama_backend_free();
        return fail("decode failed");
    }

    const llama_token sampled = llama_get_sampled_token_ith(context, -1);
    const float * sampled_logits = llama_get_sampled_logits_ith(context, -1);
    const uint32_t sampled_count = llama_get_sampled_logits_count_ith(context, -1);
    float * logits = llama_get_logits(context);
    float * logits_ith = llama_get_logits_ith(context, -1);

    bool valid = true;
    if (sampled != expected) {
        fprintf(stderr, "llama logits probe: sampled token %d, expected %d\n", sampled, expected);
        valid = false;
    }
    if (sampled_logits != NULL) {
        fprintf(stderr, "llama logits probe: sampled logits pointer is non-null\n");
        valid = false;
    }
    if (sampled_count != (uint32_t) vocab_count) {
        fprintf(stderr, "llama logits probe: sampled logits fallback count %u, expected %d\n",
                sampled_count, vocab_count);
        valid = false;
    }
    if (logits == NULL || logits_ith == NULL) {
        fprintf(stderr, "llama logits probe: full logits pointer is null\n");
        valid = false;
    }

    int32_t logits_winner = 0;
    int32_t logits_ith_winner = 0;
    if (sampled_logits == NULL && logits != NULL && logits_ith != NULL) {
        for (int32_t token = 0; token < vocab_count; ++token) {
            if (logits[token] != logits_ith[token]) {
                fprintf(stderr, "llama logits probe: logits APIs differ at token %d\n", token);
                valid = false;
                break;
            }
            if (token == expected) {
                if (logits[token] != 1.0f) {
                    fprintf(stderr, "llama logits probe: winner logit is %.9g, expected 1\n", logits[token]);
                    valid = false;
                }
            } else if (!isinf(logits[token]) || logits[token] >= 0.0f) {
                fprintf(stderr, "llama logits probe: token %d is not negative infinity\n", token);
                valid = false;
                break;
            }
            if (logits[token] > logits[logits_winner]) {
                logits_winner = token;
            }
            if (logits_ith[token] > logits_ith[logits_ith_winner]) {
                logits_ith_winner = token;
            }
        }
    }
    if (logits_winner != expected || logits_ith_winner != expected) {
        fprintf(stderr, "llama logits probe: argmax mismatch raw=%d ith=%d expected=%d\n",
                logits_winner, logits_ith_winner, expected);
        valid = false;
    }

    llama_free(context);
    llama_model_free(model);
    llama_backend_free();

    if (!valid) {
        return 1;
    }
    printf("llama logits probe: ok (vocab=%d token=%d logit=1)\n", vocab_count, expected);
    return 0;
}
