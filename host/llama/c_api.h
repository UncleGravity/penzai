#pragma once
// Full llama + ggml + chat translate-c root for the host driver (Zig module `c`).
// Only host/llama/ imports this. The backend core compiles against the ggml-only
// c_ggml.h instead, so it never sees llama.h (the .so guardrail; split deferred).

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-backend-impl.h"
#include "chat.h"
