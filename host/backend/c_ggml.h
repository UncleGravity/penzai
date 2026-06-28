#pragma once
// ggml-only translate-c root for the llama-free backend core (Zig module `c_ggml`).
// backend.zig / lower.zig / census.zig compile against this, so the core never
// sees llama.h — the .so guardrail. The full llama+ggml root is llama/c_api.h.
// (ggml-backend-impl.h supplies the vtable structs + GGML_BACKEND_API_VERSION.)

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
