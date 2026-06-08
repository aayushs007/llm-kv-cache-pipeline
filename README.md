# LLM KV Cache Pipeline & Custom Transformer

A from-scratch implementation of a 2.76M-parameter decoder-only Transformer model in PyTorch. This project focuses on optimizing autoregressive inference by replacing standard Multi-Head Attention (MHA) with advanced memory-efficient architectures.

## Core Architecture
Instead of relying on standard PyTorch `nn.Transformer` modules, this engine was built from first principles to integrate modern LLM advancements:
* **Grouped Query Attention (GQA):** Utilizes 8 Query heads and 2 Key/Value heads to compress the attention bottleneck.
* **KV Caching:** Implemented dynamic state-caching during inference, reducing memory requirements by **4x** compared to standard MHA.
* **Rotary Positional Embeddings (RoPE):** Replaced absolute sinusoidal embeddings for better sequence extrapolation.
* **Modern Activations & Norms:** Integrated **SwiGLU** feed-forward networks and **RMSNorm** for training stability.

## Model Specifications
* **Parameters:** 2.76 Million
* **Layers:** 4 
* **Embedding Dimension:** 256
* **Context Length:** 256 tokens
* **Dataset:** Tiny Shakespeare (1.0M+ tokens)

## Inference Engine
The generation script is built for dynamic text synthesis, bypassing standard greedy decoding by incorporating:
* **Temperature Scaling:** Controls the randomness of the output distribution.
* **Multinomial Sampling:** Ensures diverse and natural text generation sequences.

## Usage
Open the Jupyter Notebook (`transformer_training_pipeline.ipynb`) to view the complete pipeline, from the custom `nn.Module` classes to the training loop and the final autoregressive generation scripts.
