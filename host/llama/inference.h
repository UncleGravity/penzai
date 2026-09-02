#ifndef PENZAI_INFERENCE_V1_H
#define PENZAI_INFERENCE_V1_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PENZAI_INFERENCE_V1_ABI_VERSION 1u
#define PENZAI_INFERENCE_V1_SYMBOL "penzai.inference.v1"

enum penzai_inference_status_v1 {
    PENZAI_INFERENCE_STATUS_OK_V1 = 0,
    PENZAI_INFERENCE_STATUS_INVALID_ARGUMENT_V1 = -1,
    PENZAI_INFERENCE_STATUS_UNSUPPORTED_V1 = -2,
    PENZAI_INFERENCE_STATUS_IO_V1 = -3,
    PENZAI_INFERENCE_STATUS_DEVICE_V1 = -4,
    PENZAI_INFERENCE_STATUS_STATE_V1 = -5,
};

enum penzai_inference_feature_v1 {
    PENZAI_INFERENCE_FEATURE_GREEDY_ONLY_V1 = UINT64_C(1) << 0,
    PENZAI_INFERENCE_FEATURE_SINGLE_SEQUENCE_V1 = UINT64_C(1) << 1,
    PENZAI_INFERENCE_FEATURE_WINNER_LOGIT_V1 = UINT64_C(1) << 2,
};

enum penzai_inference_weight_format_v1 {
    PENZAI_INFERENCE_WEIGHT_NONE_V1 = 0,
    PENZAI_INFERENCE_WEIGHT_F32_V1 = 1,
    PENZAI_INFERENCE_WEIGHT_F16_V1 = 2,
    PENZAI_INFERENCE_WEIGHT_Q1_0_V1 = 3,
    PENZAI_INFERENCE_WEIGHT_Q2_0_V1 = 4,
};

enum penzai_inference_model_flag_v1 {
    // Tensor bytes are valid only for the duration of load_model().
    PENZAI_INFERENCE_MODEL_SOURCE_EPHEMERAL_V1 = UINT64_C(1) << 0,
    PENZAI_INFERENCE_MODEL_TIED_OUTPUT_V1 = UINT64_C(1) << 1,
};

enum penzai_inference_execute_flag_v1 {
    PENZAI_INFERENCE_EXECUTE_EMIT_WINNER_V1 = UINT64_C(1) << 0,
};

enum penzai_inference_result_flag_v1 {
    PENZAI_INFERENCE_RESULT_HAS_WINNER_V1 = UINT64_C(1) << 0,
};

struct penzai_inference_tensor_source_v1 {
    uint32_t struct_size;
    uint32_t format;
    uint32_t rank;
    uint32_t reserved0;
    uint64_t dimensions[4];
    const void * data;
    uint64_t data_size;
    const char * name;
    uint32_t name_size;
    uint32_t reserved1;
};

struct penzai_inference_layer_v1 {
    uint32_t struct_size;
    uint32_t reserved0;
    struct penzai_inference_tensor_source_v1 query;
    struct penzai_inference_tensor_source_v1 key;
    struct penzai_inference_tensor_source_v1 value;
    struct penzai_inference_tensor_source_v1 attention_output;
    struct penzai_inference_tensor_source_v1 gate;
    struct penzai_inference_tensor_source_v1 up;
    struct penzai_inference_tensor_source_v1 ffn_down;
    struct penzai_inference_tensor_source_v1 attention_norm;
    struct penzai_inference_tensor_source_v1 attention_q_norm;
    struct penzai_inference_tensor_source_v1 attention_k_norm;
    struct penzai_inference_tensor_source_v1 ffn_norm;
};

struct penzai_inference_model_descriptor_v1 {
    uint32_t struct_size;
    uint32_t reserved0;
    uint64_t flags;
    const char * architecture;
    uint32_t architecture_size;
    uint32_t block_count;
    uint32_t context_length;
    uint32_t embedding_length;
    uint32_t feed_forward_length;
    uint32_t attention_head_count;
    uint32_t attention_head_count_kv;
    uint32_t attention_key_length;
    uint32_t attention_value_length;
    uint32_t vocabulary_size;
    uint32_t rope_original_context;
    float rope_scaling_factor;
    float rope_frequency_base;
    float rms_epsilon;
    uint32_t reserved1;
    struct penzai_inference_tensor_source_v1 embedding;
    // Zeroed when PENZAI_INFERENCE_MODEL_TIED_OUTPUT_V1 is set.
    struct penzai_inference_tensor_source_v1 output;
    struct penzai_inference_tensor_source_v1 output_norm;
    const struct penzai_inference_layer_v1 * layers;
    uint32_t layer_count;
    uint32_t reserved2;
};

struct penzai_inference_model_result_v1 {
    uint32_t struct_size;
    uint32_t reserved0;
    void * model;
    uint32_t max_context_tokens;
    uint32_t max_tile_tokens;
    uint64_t features;
};

struct penzai_inference_session_params_v1 {
    uint32_t struct_size;
    uint32_t capacity_tokens;
    uint64_t flags;
};

struct penzai_inference_execute_request_v1 {
    uint32_t struct_size;
    uint32_t first_position;
    uint32_t token_count;
    uint32_t reserved0;
    uint64_t flags;
    const uint32_t * token_ids;
};

struct penzai_inference_execute_result_v1 {
    uint32_t struct_size;
    uint32_t committed_tokens;
    uint64_t flags;
    uint32_t token_id;
    float winning_logit;
    uint64_t device_cycles;
    uint64_t device_time_ns;
};

struct penzai_inference_v1 {
    uint32_t abi_version;
    uint32_t struct_size;
    uint64_t features;
    // On any status, a non-null result->model transfers ownership to the
    // caller and will be released exactly once with unload_model().
    int32_t (*load_model)(
        void * device,
        const struct penzai_inference_model_descriptor_v1 * descriptor,
        struct penzai_inference_model_result_v1 * result);
    void (*unload_model)(void * model);
    // On any status, a non-null *session transfers ownership to the caller
    // and will be released exactly once with close_session().
    int32_t (*open_session)(
        void * model,
        const struct penzai_inference_session_params_v1 * params,
        void ** session);
    void (*close_session)(void * session);
    int32_t (*reset_session)(void * session);
    int32_t (*execute)(
        void * session,
        const struct penzai_inference_execute_request_v1 * request,
        struct penzai_inference_execute_result_v1 * result);
    const char * (*status_message)(int32_t status);
    void * reserved[8];
};

typedef const struct penzai_inference_v1 * (*penzai_inference_query_v1_fn)(void);

#ifdef __cplusplus
}
#endif

#endif
