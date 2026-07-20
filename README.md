# MA2288 Research V2: Lossless Latent Self-Speculative Decoding

This repository contains the experiments for an MA2288 research project on
next-latent prediction and self-speculative decoding. The project begins with a
small, controlled DistilGPT-2 study and progresses from latent-state prediction
to a complete lossless speculative decoding prototype with KV-cache support.

The central question is:

> Can a lightweight model predict several future hidden states accurately
> enough to reduce target-model decoding work and produce real end-to-end
> inference speedup?

The experiments show that better latent prediction and lower decoder-output KL
divergence do not automatically translate into longer accepted speculative
prefixes or higher wall-clock throughput. On the current DistilGPT-2/T4 setup,
the prototype reduces the number of target-model calls but remains slower than
vanilla decoding because verification, cache update, and sequential drafting
cost dominate.

## Research questions

This project studies five questions:

1. Can a small transition model predict the next hidden state of a frozen
   language model?
2. Does multi-step training reduce long-horizon latent and output-space error?
3. Do decoder-aware and acceptance-aware objectives improve speculative-token
   acceptance?
4. Can these improvements produce a correct, lossless self-speculative decoder?
5. Under what acceptance and runtime conditions can the system become faster
   than vanilla autoregressive decoding?

## Experimental setup

- Target model: `distilgpt2`
- Hidden dimension: 768
- Dataset: WikiText-derived token sequences
- Sequence length: 64 for the original latent-prediction experiments
- Prompt length: 32 for the Research V2 rollout experiments
- Transition model: residual MLP with a 512-dimensional bottleneck
- Main hardware: NVIDIA Tesla T4
- Framework: PyTorch and Hugging Face Transformers
- Evaluation horizons: 1, 2, 4, 8, 16, and 32 where applicable
- Speculative horizons: 2, 4, and 8

The target language model is frozen throughout the experiments. Only the
lightweight latent transition model is trained.

## Repository structure

```text
notebooks/
  01_environment_check.ipynb
  02_extract_hidden_states.ipynb
  03_prepare_dataset.ipynb
  04_identity_baseline.ipynb
  05_train_one_step_mlp.ipynb
  06_evaluate_rollout.ipynb
  07_train_multistep_mlp.ipynb
  08_compare_all_models.ipynb
  09_periodic_refresh.ipynb
  10_bootstrap_analysis.ipynb
  11_v2_acceptance_setup.ipynb
  12_v2_single_round_acceptance.ipynb
  13_v2_acceptance_predictors.ipynb
  14_v2_collect_onpolicy_rollouts.ipynb
  15_v2_train_onpolicy_latent.ipynb
  16_v2_evaluate_onpolicy_acceptance.ipynb
  17_survival_weighted_training.ipynb
  18_fresh_acceptance_benchmark.ipynb
  19_acceptance_objectives.ipynb
  20_lossless_self_speculative_decoding.ipynb

results/
  figures/
  tables/

checkpoints/
  one_step_transition_seed42.pt
  multistep_transition_seed42.pt

checkpoints_v2/
  onpolicy_latent_transition_seed42.pt
  survival_weighted_transition_seed42.pt

checkpoints_v3/
  decoder_kl_c2_selected_seed42.pt
  local_lk_c2_selected_seed42.pt
  survival_lk_c2_selected_seed42.pt

data_v2/
  onpolicy_rollouts_seed42.pt

data_v3/
  notebook19_prompt_splits_seed42.pt
```

Notebooks 11–16 retain their existing filenames in the repository; the wildcard
entries above represent the intermediate Research V2 experiments. Large
datasets and checkpoints may be excluded from GitHub and stored separately.

## Method

### Residual latent transition

Let $h_t \in \mathbb{R}^{768}$ be the frozen target model's hidden state at
position $t$, and let $e(x_{t+1})$ be the embedding of the next token. The
transition model predicts the next hidden state using

```math
\widehat{h}_{t+1}
=
h_t + f_\theta\!\left(
\mathrm{LN}\!\left([h_t; e(x_{t+1})]\right)
\right),
```

where $[\,\cdot\,;\,\cdot\,]$ denotes concatenation and $f_\theta$ is a
two-layer MLP with a GELU activation and a 512-dimensional bottleneck.

### Latent prediction loss

The normalized hidden-state prediction error is

```math
\mathcal{L}_{\mathrm{latent}}
=
\frac{
\left\lVert \widehat{h}_{t+1}-h_{t+1}\right\rVert_2^2
}{
\left\lVert h_{t+1}\right\rVert_2^2+\varepsilon
}.
```

For a rollout of length $K$, the recurrent transition is applied repeatedly
and the loss is averaged across predicted steps.

### Decoder-aware output loss

Hidden-state similarity is not sufficient for speculative decoding. Let
$W_{\mathrm{LM}}$ denote the frozen language-model output head. The target and
draft distributions are

```math
p_t = \mathrm{softmax}\!\left(W_{\mathrm{LM}}h_t\right),
\qquad
q_t = \mathrm{softmax}\!\left(W_{\mathrm{LM}}\widehat{h}_t\right).
```

The decoder-aware objective uses

```math
\mathcal{L}_{\mathrm{KL}}
=
D_{\mathrm{KL}}\!\left(p_t\,\|\,q_t\right).
```

### Speculative acceptance probability

For a token $y_t \sim q_t$, standard stochastic speculative decoding accepts
the draft token with probability

```math
a_t
=
\min\!\left(1,\frac{p_t(y_t)}{q_t(y_t)}\right).
```

If a token is rejected, the replacement token is sampled from the normalized
positive residual distribution

```math
r_t(x)
=
\frac{[p_t(x)-q_t(x)]_+}
{\sum_v [p_t(v)-q_t(v)]_+},
```

where $[z]_+=\max(z,0)$. This correction, together with target verification,
makes the stochastic decoder distributionally equivalent to vanilla target
decoding up to numerical precision. In this project, **lossless** refers to this
correctness property; it is a requirement of the system rather than the primary
algorithmic contribution.

### Sequence-survival objective

The probability that a speculative prefix survives through step $k$ is

```math
S_k = \prod_{j=1}^{k} a_j.
```

The expected number of accepted draft tokens beyond the first position can be
written as

```math
\mathbb{E}[A_{>1}]
=
\sum_{k=2}^{K} S_k.
```

This motivates a survival-weighted training objective that emphasizes errors
before the first rejection. The experiments compare latent, decoder-KL,
token-local acceptance, and sequence-survival objectives under a controlled
training protocol.

## Research V1: latent prediction

### One-step transition

The one-step residual MLP improves validation hidden-state prediction over the
identity baseline:

- Best validation normalized L2: 0.2063
- Best validation cosine similarity: 0.9725
- Identity normalized L2 at horizon 1: 0.3954
- One-step normalized L2 at horizon 1: 0.1985

At horizon 1, the one-step model reduces normalized latent error by 49.8% and
output KL by 71.1% relative to the identity baseline.

### Multi-step transition

Multi-step training provides its largest gains at long horizons. Relative to the
one-step model, it reduces normalized latent error by 13.3% and output KL by
37.8% at horizon 32.

| Horizon | One-step normalized L2 | Multi-step normalized L2 | One-step output KL | Multi-step output KL |
|---:|---:|---:|---:|---:|
| 1 | 0.1985 | 0.1952 | 1.3296 | 1.3117 |
| 4 | 0.2052 | 0.1985 | 1.7668 | 1.6811 |
| 8 | 0.2485 | 0.2402 | 2.1488 | 1.9959 |
| 16 | 0.2778 | 0.2514 | 2.4336 | 2.0055 |
| 32 | 0.3394 | 0.2943 | 3.7030 | 2.3019 |

Bootstrap analysis confirms that the output-KL improvement is positive at every
evaluated horizon. The estimated reduction grows from 0.0179 at horizon 1 to
1.4010 at horizon 32.

### Periodic refresh

Refreshing the latent state with the target model reduces accumulated rollout
error. More frequent refresh gives better approximation quality but performs
more target computation. This establishes the core systems trade-off between
prediction error and target-model work.

## Research V2: from proxy metrics to acceptance

### On-policy rollout training

Teacher-forced latent accuracy does not fully describe the states encountered
during autoregressive drafting. Research V2 therefore constructs fresh
on-policy rollouts from the transition model and fine-tunes on the resulting
state distribution.

On-policy latent fine-tuning reduces validation rollout loss by 2.93%, but its
fresh acceptance improvement is small. The mean empirical acceptance changes
from 0.3750 for the multi-step model to 0.3761 for the on-policy model.

### Survival-weighted decoder training

Survival-weighted fine-tuning gives the following validation improvements:

- Combined loss reduction: 3.78%
- Latent loss reduction: 3.71%
- Unweighted output-KL reduction: 6.42%
- Survival-weighted output-KL reduction: 5.02%

However, lower average KL does not produce a statistically reliable increase in
fresh accepted prefix length. This is an important negative result: improving a
smooth proxy metric is not enough when system utility is determined by the
first rejection event.

### Direct acceptance objectives

Notebook 19 compares three fine-tuning objectives:

- Decoder-KL
- Token-local likelihood/acceptance loss (Local-LK)
- Sequence-survival likelihood loss (Survival-LK)

After gradient calibration and a corrected checkpoint-selection protocol, all
three objectives improve their validation estimate of accepted tokens beyond
the first position:

| Objective | Baseline expected beyond first | Best expected beyond first | Selected epoch |
|---|---:|---:|---:|
| Decoder-KL | 0.7379 | 0.7638 | 4 |
| Local-LK | 0.7379 | 0.7666 | 2 |
| Survival-LK | 0.7379 | 0.7687 | 7 |

Fresh test-time sampling does not show a stable advantage of Survival-LK over
Local-LK. The expected and empirical differences are close to zero or negative,
and the bootstrap confidence intervals cross zero. The additional
sequence-survival objective is therefore not supported as the main method.

## Lossless self-speculative decoding prototype

Notebook 20 implements a complete batch-size-one prototype with:

- recurrent latent drafting;
- target-model verification;
- standard stochastic acceptance;
- rejection-residual correction;
- bonus-token sampling when all drafts are accepted;
- KV-cache growth and rollback; and
- greedy and stochastic correctness tests.

Cached and full recomputation agree within low-precision numerical tolerance:

- Maximum hidden-state difference: $1.18\times10^{-4}$
- Maximum logit difference: $1.11\times10^{-4}$
- Maximum probability difference: $8.94\times10^{-7}$
- Top-1 agreement: true

All greedy speculative outputs exactly match vanilla greedy decoding in the
correctness tests. The stochastic smoke test also passes.

## End-to-end benchmark

The benchmark uses 10 held-out prompts, 64 generated tokens, three timing
repetitions, batch size 1, and an NVIDIA Tesla T4. It measures decode-only
throughput after warm-up.

Representative best results:

| Sampling mode | Best configuration | Mean speculative/vanilla speed ratio | 95% bootstrap CI | Result |
|---|---|---:|---:|---|
| Greedy | Multi-step, horizon 8 | 0.720 | [0.611, 0.837] | Slower |
| Stochastic | Decoder-KL, horizon 4 | 0.618 | [0.602, 0.635] | Slower |

A speed ratio below 1 means the speculative decoder is slower than vanilla
decoding. No tested configuration achieves end-to-end speedup.

Speculation reduces target-model calls per emitted token, but target-call count
alone is not a sufficient measure of efficiency. A verification call processes
multiple positions, and the implementation also incurs sequential draft and
cache-update costs.

## Component profiling

| Method | Horizon | Draft latency (ms) | Verification latency (ms) | Update latency (ms) | Estimated round latency (ms) |
|---|---:|---:|---:|---:|---:|
| Multi-step | 2 | 0.440 | 3.120 | 1.259 | 4.819 |
| Multi-step | 4 | 0.867 | 3.166 | 1.260 | 5.294 |
| Multi-step | 8 | 1.721 | 3.143 | 1.270 | 6.134 |
| Decoder-KL | 2 | 0.440 | 3.127 | 1.259 | 4.827 |
| Decoder-KL | 4 | 0.867 | 3.157 | 1.258 | 5.282 |
| Decoder-KL | 8 | 1.721 | 3.154 | 1.271 | 6.146 |
| Local-LK | 2 | 0.440 | 3.133 | 1.261 | 4.835 |
| Local-LK | 4 | 0.867 | 3.199 | 1.268 | 5.334 |
| Local-LK | 8 | 1.721 | 3.168 | 1.272 | 6.162 |

Vanilla one-token update latency is approximately 1.186 ms.

If $L_v$ is vanilla latency per token, $L_d(K)$ is draft latency,
$L_{\mathrm{verify}}(K)$ is verification latency, and
$L_{\mathrm{update}}$ is cache-update latency, then a speculative round is
faster only if

```math
\frac{
L_d(K)+L_{\mathrm{verify}}(K)+L_{\mathrm{update}}
}{
\mathbb{E}[N_{\mathrm{emit}}]
}
< L_v.
```

Equivalently, the required mean number of emitted tokens per round is

```math
\mathbb{E}[N_{\mathrm{emit}}]
>
\frac{
L_d(K)+L_{\mathrm{verify}}(K)+L_{\mathrm{update}}
}{L_v}.
```

Using the measured Multi-step latencies, the approximate break-even conditions
are:

| Horizon | Required emitted tokens per round | Interpretation |
|---:|---:|---|
| 2 | 4.06 | Impossible because at most 3 tokens can be emitted |
| 4 | 4.46 | Requires nearly complete draft acceptance |
| 8 | 5.17 | Requires more than about 4.17 accepted drafts on average |

The dominant measured costs are the roughly 3.1 ms verification pass and the
roughly 1.26 ms cache-update pass. Sequential drafting also scales almost
linearly with horizon. These costs explain why lower target-call counts do not
produce speedup.

## Main findings

1. **One-step latent accuracy does not predict long-horizon functional
   quality.** A transition can have good local hidden-state similarity while its
   decoded token distribution deteriorates over a rollout.
2. **Multi-step training improves long-horizon output quality.** The benefit is
   clearest in output KL at horizons 16 and 32.
3. **Better output KL does not guarantee better accepted prefix length.** The
   first-rejection structure of speculative decoding creates a mismatch between
   average differentiable losses and actual sequence utility.
4. **Direct acceptance tuning gives only small gains with the current residual
   MLP.** Survival-LK is not reliably better than the simpler Local-LK baseline.
5. **Acceptance is necessary but not sufficient for speedup.** Verification,
   cache maintenance, draft execution, batching, and hardware utilization must
   be included in the objective and evaluation.
6. **The current residual MLP should not be scaled as the main architecture.**
   The next phase should first reproduce a strong public speculator and runtime
   baseline before proposing another loss.

## Relation to NextLat

This project is inspired by NextLat, which studies next-latent prediction as a
compact world model and reports acceleration using a substantially stronger
training and systems setup. The present repository is an independent small-scale
study rather than a reproduction of the full NextLat system.

The difference matters: this project uses DistilGPT-2, a small residual MLP,
limited data, a single T4 GPU, sequential latent rollout, and an unoptimized
batch-size-one Python prototype. Its negative throughput result does not
contradict acceleration reported by NextLat. Instead, it isolates the conditions
that a practical implementation must satisfy.

## Limitations

- Only one small target model is evaluated.
- The principal benchmark uses one Tesla T4.
- The decoder supports batch size 1 rather than continuous or dynamic batching.
- The recurrent drafter is sequential and does not exploit parallel token
  prediction.
- The experiment scale is too small for broad model, dataset, and hardware
  generalization claims.
- Several comparisons use one training seed; stronger claims require multiple
  seeds and fresh held-out prompts.
- The implementation is a research prototype rather than an optimized vLLM or
  SGLang kernel path.

## Reproduction

1. Open the notebooks in numerical order.
2. Run notebooks 01–10 for the original latent-prediction study.
3. Run the Research V2 data and on-policy notebooks before notebooks 17–20.
4. Run notebook 19 to create the selected acceptance-objective checkpoints.
5. Run notebook 20 for correctness, end-to-end timing, bootstrap analysis, and
   component profiling.
6. Export result tables to `results/tables/` and figures to
   `results/figures/`.

For timing experiments, use the same GPU, package versions, prompt set,
generation length, warm-up procedure, and number of repetitions. Do not compare
raw Colab timings collected under different runtime conditions.

Large `.pt` artifacts are not required in the GitHub release. Record their
creation notebook, random seed, model name, and expected path so that they can be
regenerated.

## Next-stage plan

The next stage should be a bounded strong-baseline pilot rather than additional
loss tuning on the current MLP:

1. Reproduce a public strong speculator such as an EAGLE-family or parallel
   direct-token drafter on a larger target model.
2. Integrate it with an optimized inference runtime such as vLLM or SGLang.
3. Measure acceptance, accepted prefix, target tokens processed, throughput,
   time per output token, time to first token, and memory together.
4. Vary prompt domain, context length, concurrency, speculative horizon, model,
   and GPU.
5. Profile verification, draft execution, cache update, synchronization, and
   scheduling costs.
6. Only after reproducing a real speedup, test a cost-aware dynamic-horizon or
   runtime-aware utility method.

A reasonable pilot is one A100 80 GB GPU for at most 48 GPU-hours. Larger-scale
training should be requested only if the pilot reproduces a stable speedup or
reveals a clear, repeatable architecture/runtime bottleneck.

## Conclusion

This project successfully progresses from hidden-state prediction to a complete
lossless self-speculative decoding prototype. It also establishes a useful
negative systems result: better latent metrics, lower output KL, and fewer
target calls do not by themselves imply faster inference.

The strongest conclusion is therefore not that the current method accelerates
DistilGPT-2, but that speculative latent decoding must optimize sequence
acceptance and real runtime cost jointly. Future work should evaluate this
hypothesis using a stronger speculator, a larger target model, and a
production-oriented inference runtime.

## Reference

This project is inspired by the following work:

Jayden Teoh, Manan Tomar, Kwangjun Ahn, Edward S. Hu, Tim Pearce,
Pratyusha Sharma, Akshay Krishnamurthy, Riashat Islam, Alex Lamb,
and John Langford.

**Next-Latent Prediction Transformers Learn Compact World Models.**
2026.

- Paper: <https://arxiv.org/abs/2511.05963>
- Official implementation: <https://github.com/JaydenTeoh/NextLat>

## License and acknowledgement

This repository is an academic research project. Please consult the licenses of
the datasets, pretrained models, and external implementations before reuse.
The NextLat authors are not responsible for this independent implementation or
its experimental conclusions.
