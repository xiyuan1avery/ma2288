# V2: Risk-Aware Latent Speculative Decoding

## Central hypothesis

One-step latent accuracy is insufficient to predict long-horizon
functional quality. Decoder-aware rollout risk should predict
speculative acceptance and enable dynamic draft horizons.

## Gate 1: Acceptance prediction

Questions:

1. Does output KL predict token acceptance?
2. Is output KL more predictive than latent L2, cosine similarity,
   entropy, or top-1 margin?
3. Does multi-step training improve acceptance primarily at later
   draft positions?

Success criteria:

- Output KL is negatively associated with acceptance.
- Multi-step transitions improve later-position acceptance.
- The improvement is larger at long horizons than at short horizons.

## Gate 2: Oracle adaptive horizon

Question:

Can an oracle dynamic horizon outperform the best validation-tuned
fixed horizon?

Success criteria:

- The best horizon varies substantially across contexts.
- Oracle adaptation improves accepted tokens per target verification.

## Gate 3: Deployable risk predictor

Candidate features:

- draft entropy;
- top-1/top-2 probability margin;
- latent residual norm;
- rollout depth;
- accumulated predicted risk.

Success criteria:

- The learned predictor is calibrated.
- It improves over entropy-only and fixed-horizon policies.
- It recovers a substantial fraction of oracle performance.

## Gate 4: End-to-end system

Metrics:

- acceptance by draft position;
- accepted tokens per verification;
- target calls per generated token;
- latency per generated token;
- throughput;
- peak GPU memory;
- policy overhead.

The final system must preserve the target sampling distribution.
