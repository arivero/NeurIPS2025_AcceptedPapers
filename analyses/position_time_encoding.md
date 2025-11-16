# Position and Time Encoding Papers (NeurIPS 2025)

The following papers introduce new positional or temporal encoding ideas.

## Causality-Induced Positional Encoding for Transformer-Based Representation Learning of Non-Sequential Features
- **Idea:** CAPE infers a causal DAG over unordered features, embeds it in hyperbolic space to preserve causal properties, and converts the result into rotary positional encodings for self-attention.
- **Why interesting:** Provides positional signals for non-sequential but causally related inputs, giving robustness to positional disturbances and improved performance on synthetic and real datasets.

## Physics-Informed Position Encoding (PIPE) for Alignment of Satellite Images and Time Series
- **Idea:** Adds physics-aware positional indexing plus a variant-frequency positional encoding to inject geospatial and temporal physics into VLM embeddings for satellite imagery and time series forecasting.
- **Why interesting:** Improves multimodal alignment and typhoon intensity forecasting by preserving physical and sequential order information.

## LEDiT: Length-Extrapolatable Diffusion Transformer without Positional Encoding
- **Idea:** Removes explicit positional encodings (e.g., RoPE) and relies on causal attention with an added locality module so diffusion transformers can extrapolate to much longer image resolutions.
- **Why interesting:** Demonstrates strong resolution scaling (up to 4×) without explicit PEs, suggesting causal attention can implicitly encode positions.

## Toward Relative Positional Encoding in Spiking Transformers
- **Idea:** Introduces Gray-PE and Log-PE to approximate relative positional encodings while keeping binary spike representations, with 2D extensions for images.
- **Why interesting:** Enables relative positional information in spiking Transformers, boosting tasks like time-series forecasting and image classification.

## Adaptive Time Encoding for Irregular Multivariate Time-Series Classification
- **Idea:** Learns latent representations at adaptive reference points to capture missingness patterns and irregular intervals, with consistency regularization to fuse temporal and inter-variable cues.
- **Why interesting:** Addresses uneven sampling in multivariate time series, improving classification accuracy with efficient computation.

## PaTH Attention: Position Encoding via Accumulating Householder Transformations
- **Idea:** Uses data-dependent Householder transformations accumulated along the sequence to produce expressive, input-aware position encodings compatible with FlashAttention-style training.
- **Why interesting:** Outperforms RoPE and other baselines on synthetic and language modeling tasks by making positional information depend on the actual tokens.

## Rethinking Scale-Aware Temporal Encoding for Event-based Object Detection
- **Idea:** A CNN–RNN hybrid with decoupled deformable recurrent layers models temporal dynamics at lower spatial scales, enabling multi-scale temporal feature extraction before fusion.
- **Why interesting:** Shows better event-based detection accuracy, emphasizing early, scale-aware temporal encoding beyond attention-centric designs.

## Projective Positional Encoding for Multiview Transformers (PRoPE)
- **Idea:** Encodes full projective camera relationships (intrinsics + extrinsics) to condition multi-view transformers, remaining invariant to global frame choices.
- **Why interesting:** Improves robustness to varying camera parameters across novel view synthesis, stereo depth, and spatial cognition tasks.
