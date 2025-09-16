# Loop Attention Implementation for llama.cpp

This document describes the loop attention feature implementation for llama.cpp.

## Overview

Loop attention allows LLMs to process selected layers multiple times in a single inference pass, enabling more sophisticated attention patterns and potentially improved performance for certain tasks.

## Implementation Details

### API Changes

#### New Context Parameters
Added to `llama_context_params` structure:
- `loop_attention_enabled`: Boolean to enable/disable loop attention
- `loop_attention_n_loops`: Number of iterations through selected layers (default: 1)
- `loop_attention_start_layer`: Starting layer for looping (0-based, default: 0)
- `loop_attention_end_layer`: Ending layer for looping (-1 = last layer, default: -1)

#### Command Line Interface
New CLI arguments for llama-cli:
- `--loop-attention`: Enable loop attention processing
- `--loop-attention-loops N`: Set number of loops (default: 1)
- `--loop-attention-start N`: Set starting layer (default: 0)
- `--loop-attention-end N`: Set ending layer (default: -1)

### Architecture Support

Currently implemented for:
- LLaMA architecture (`llm_build_llama`)

### Core Implementation

The implementation modifies the layer processing loop in the model builder:

1. **First iteration**: Processes all layers (0 to n_layer-1) normally
2. **Subsequent iterations**: Processes only the specified loop range (start_layer to end_layer)
3. **Data flow**: Stores intermediate layer outputs to enable proper data flow between iterations
4. **Memory management**: Uses `std::vector<ggml_tensor *>` to store intermediate results

### Example Usage

```bash
# Enable loop attention with 3 loops through layers 5-10
./llama-cli --loop-attention --loop-attention-loops 3 --loop-attention-start 5 --loop-attention-end 10

# Loop through all layers except the first and last
./llama-cli --loop-attention --loop-attention-loops 2 --loop-attention-start 1 --loop-attention-end -2
```

## Technical Implementation

### File Modifications
- `include/llama.h`: Added parameters to public API
- `src/llama-cparams.h`: Added parameters to internal context structure  
- `src/llama-context.cpp`: Added default values and parameter transfer
- `src/llama-model.cpp`: Implemented core loop logic in LLaMA builder
- `common/common.h`: Added CLI parameter definitions
- `common/common.cpp`: Added parameter conversion
- `common/arg.cpp`: Added command-line argument parsing

### Key Functions Modified
- `llama_context_default_params()`: Added default values
- `common_context_params_to_llama()`: Added parameter conversion
- `llm_build_llama()`: Core loop attention implementation

## Current Status

✅ **Completed:**
- Basic implementation for LLaMA architecture
- CLI interface
- Parameter validation
- Build system integration

🔄 **Next Steps:**
- Testing with actual models
- Performance evaluation
- Extension to other architectures (Falcon, Baichuan, etc.)
- Documentation and examples

## Architecture Extension

To extend loop attention to other model architectures, modify the corresponding builder functions in `src/llama-model.cpp`:
- `llm_build_falcon`
- `llm_build_baichuan` 
- `llm_build_grok`
- etc.

The implementation pattern follows the same structure as `llm_build_llama`.