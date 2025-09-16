#pragma once

#include <cstdint>

// Loop attention configuration parameters
struct llama_loop_attention_params {
    bool enabled = false;                 // Enable loop attention
    int32_t n_loops = 1;                 // Number of times to loop through layers
    int32_t loop_start_layer = 0;        // Starting layer for loop attention (0-based)
    int32_t loop_end_layer = -1;         // Ending layer for loop attention (-1 = last layer)
};

// Default loop attention parameters
inline llama_loop_attention_params llama_loop_attention_default_params() {
    llama_loop_attention_params result = {};
    result.enabled = false;
    result.n_loops = 1;
    result.loop_start_layer = 0;
    result.loop_end_layer = -1;
    return result;
}