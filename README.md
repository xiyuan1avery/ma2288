# Next-Latent Stability Experiment

This project empirically studies error accumulation in recursively
predicted language-model hidden states.

## Research questions

1. Does low one-step latent prediction error guarantee stable multi-step rollout?
2. How does latent error relate to output-distribution KL divergence?
3. Can multi-step supervision and periodic state refresh reduce long-horizon error?

## Experimental setup

- Frozen pretrained language model: DistilGPT-2
- Trainable component: residual latent transition MLP
- Environment: Google Colab
- Main metrics: normalized latent error, cosine similarity, output KL,
  and top-1 token agreement

## Reference

This project is inspired by:

Jayden Teoh et al.,
"Next-Latent Prediction Transformers Learn Compact World Models," 2026.

Official implementation:
https://github.com/JaydenTeoh/NextLat
