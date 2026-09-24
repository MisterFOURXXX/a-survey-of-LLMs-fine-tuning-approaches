# A Survey of LLM Fine-Tuning Approaches

> An end-to-end, reproducible research repository that implements, compares, and
> evaluates **4 types consist of modern fine-tuning, preference-optimization, reward-modeling,
> and reinforcement-learning algorithms** for Large Language Models — all built
> on a single `google/gemma-3-270m` backbone and driven from YAML configs.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.10%2B-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/transformers-4.57-yellow.svg)](https://huggingface.co/docs/transformers)
[![TRL](https://img.shields.io/badge/trl-0.14.0-green.svg)](https://huggingface.co/docs/trl)
[![PEFT](https://img.shields.io/badge/peft-0.19-orange.svg)](https://huggingface.co/docs/peft)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Foundations](#2-foundations)
   - [2.1 Supervised Fine-Tuning (SFT)](#21-supervised-fine-tuning-sft)
   - [2.2 LoRA — Low-Rank Adaptation](#22-lora--low-rank-adaptation)
   - [2.3 QLoRA — 4-bit Quantized LoRA](#23-qlora--4-bit-quantized-lora)
   - [2.4 DPO — Direct Preference Optimization](#24-dpo--direct-preference-optimization)
   - [2.5 ORPO — Odds-Ratio Preference Optimization](#25-orpo--odds-ratio-preference-optimization)
   - [2.6 KTO — Kahneman-Tversky Optimization](#26-kto--kahneman-tversky-optimization)
   - [2.7 IPO — Identity Preference Optimization](#27-ipo--identity-preference-optimization)
   - [2.8 CPO — Contrastive Preference Optimization](#28-cpo--contrastive-preference-optimization)
   - [2.9 Reward Modeling — Bradley-Terry](#29-reward-modeling--bradley-terry)
   - [2.10 Plackett-Luce Reward Modeling](#210-plackett-luce-reward-modeling)
   - [2.11 GRPO — Group Relative Policy Optimization](#211-grpo--group-relative-policy-optimization)
   - [2.12 RLOO — REINFORCE Leave-One-Out](#212-rloo--reinforce-leave-one-out)
   - [2.13 Online DPO](#213-online-dpo)
   - [2.14 Nash-MD — Nash Mirror Descent](#214-nash-md--nash-mirror-descent)
   - [2.15 XPO — Exploratory Preference Optimization](#215-xpo--exploratory-preference-optimization)
   - [2.16 Knowledge Distillation — GKD](#216-knowledge-distillation--gkd-generative-knowledge-distillation)
   - [2.17 MiniLLM](#217-minillm)
   - [2.18 Meta-Learning — MAML, Reptile, Meta-ICL](#218-meta-learning--maml-reptile-meta-icl)
3. [Evaluation](#3-evaluation)
   - [3.1 Training Metrics (per method)](#31-training-metrics-per-method)
     - [3.1.1 SFT (and SFT+LoRA, SFT+QLoRA)](#311-sft-and-sftlora-sftqlora)
     - [3.1.2 DPO (and its variants: IPO, ORPO, CPO, KTO)](#312-dpo-and-its-variants-ipo-orpo-cpo-kto)
     - [3.1.3 Reward Modeling (Bradley-Terry, Plackett-Luce)](#313-reward-modeling-bradley-terry-plackett-luce)
     - [3.1.4 GRPO (Group Relative Policy Optimization)](#314-grpo-group-relative-policy-optimization)
     - [3.1.5 RLOO, Online DPO, Nash-MD, XPO (Online RL Family)](#315-rloo-online-dpo-nash-md-xpo-online-rl-family)
     - [3.1.6 Knowledge Distillation (GKD, MiniLLM)](#316-knowledge-distillation-gkd-minillm)
     - [3.1.7 Meta-Learning (MAML, Reptile, Meta-ICL)](#317-meta-learning-maml-reptile-meta-icl)
   - [3.2 Evaluation Metrics](#32-evaluation-metrics)
     - [3.2.1 BLEU (Bilingual Evaluation Understudy)](#321-bleu-bilingual-evaluation-understudy)
     - [3.2.2 ROUGE-L](#322-rouge-l-recall-oriented-understudy-for-gisting-evaluation--longest-common-subsequence)
     - [3.2.3 Exact Match (EM)](#323-exact-match-em)
     - [3.2.4 Length Statistics (Diagnostic)](#324-length-statistics-diagnostic)
4. [Repository Structure](#4-repository-structure)
5. [Environment Setup (Step-by-Step)](#5-environment-setup-step-by-step)
   - [5.1 Clone the repository](#51-clone-the-repository)
   - [5.2 Create a Python environment](#52-create-a-python-environment)
   - [5.3 Install pinned dependencies](#53-install-pinned-dependencies)
   - [5.4 Set up the Kaggle API credentials](#54-set-up-the-kaggle-api-credentials)
   - [5.5 Hugging Face access (optional)](#55-hugging-face-access-optional)
6. [Dataset Preparation](#6-dataset-preparation)
   - [6.1 Download](#61-download)
   - [6.2 Raw schema](#62-raw-schema)
   - [6.3 Cleaning pipeline](#63-cleaning-pipeline-in-srcdatapreprocesspy)
   - [6.4 Why splits are done on the DataFrame](#64-why-splits-are-done-on-the-dataframe-not-on-the-dataset)
7. [Configuration System](#7-configuration-system)
   - [7.1 Example — `configs/base.yaml`](#71-example--configsbaseyaml)
   - [7.2 Example — `configs/sft_lora.yaml`](#72-example--configssft_lorayaml)
   - [7.3 Loading programmatically](#73-loading-programmatically)
8. [Running Experiments](#8-running-experiments)
   - [8.1 CLI (recommended for reproducibility)](#81-cli-recommended-for-reproducibility)
   - [8.2 From a notebook](#82-from-a-notebook)
   - [8.3 Output artefacts](#83-output-artefacts)
9. [Citation & References](#9-citation--references)
10. [License & Acknowledgements](#license--acknowledgements)

---

## 1. Project Overview

This repository is a **research-grade empirical study** of modern
post-training techniques for Large Language Models. Every method is
implemented as a thin, uniform wrapper around the `transformers`, `trl`,
and `peft` libraries, so the **only thing that differs between two
experiments is a single YAML file**.

All experiments use:

| Component | Choice | Rationale |
|---|---|---|
| Base model | `google/gemma-3-270m` | Small enough to run on a single consumer GPU (T4, RTX 3060+), large enough to exhibit real fine-tuning dynamics. |
| Dataset | Kaggle **StackSample** (StackOverflow Q&A) | Real, noisy, license-friendly, and supports every downstream task (SFT, preference pairs, reward pairs, GRPO prompts). |
| Framework | `transformers==4.57`, `trl==0.14.0`, `peft==0.19` | Pinned for reproducibility across TRL's breaking renames (`max_seq_length` -> `max_length`, `top_p` in GRPO, etc.). |
| Orchestration | YAML configs + `scripts/train_*.py` | Every hyperparameter lives in one file; no magic numbers in code. |

**Methods covered** (all four trainer families):

- **Supervised Fine-Tuning:** SFT, SFT+LoRA, SFT+QLoRA
- **Preference Optimization:** DPO, ORPO, KTO, CPO, IPO
- **Reward Modeling:** Bradley-Terry, Plackett-Luce, Ensembles, Multi-Objective, Distillation
- **Online RL:** GRPO, RLOO, Online DPO, Nash-MD, XPO
- **PEFT:** LoRA, QLoRA, AdaLoRA, DoRA, VeRA, Prefix/Prompt/P-Tuning, Adapters, LoReFT, Bone, SSF, MAM
- **Distillation & Meta-Learning:** GKD, MiniLLM, Multi-Teacher, Self-Distillation, CRD, TAKD, MAML, Reptile, Meta-ICL

---

## 2. Foundations

This section gives the conceptual foundation, step-by-step mechanics, and mathematical formulations for every method implemented in the repo. For each method, we state the core idea, how it operates, what is trained versus frozen, mathematical objectives, and relevant configurations or trainers.

### 2.1 Supervised Fine-Tuning (SFT)

**Core Concept & Mechanics.** SFT adapts a pre-trained base model to follow instructions or imitate a specific target style. It teaches the model how to speak like an assistant, but it does not teach it which response is better when multiple valid answers exist.

1. **Data Preparation:** Inputs are formatted into sequences containing a prompt (instruction) $x$ and a completion (target response) $y$.
2. **Masked Loss Calculation:** The entire sequence passes through the model, but loss is computed only on the target response tokens (`labels = -100` on the prompt), ensuring the model is not penalized or rewarded for predicting the user's prompt text.
3. **Distribution Matching:** The model adjusts its parameters $\theta$ to maximize the probability of outputting the exact ground-truth tokens in sequence by minimizing the negative log-likelihood (NLL):

$$
\mathcal{L}_{\text{SFT}}(\theta) = - \sum_{t=1}^{T} \log p_\theta(y_t \mid x, y_{<t})
$$

* **Configs:** `configs/sft_lora.yaml`, `configs/sft_qlora.yaml`
* **Trainer:** `trl.SFTTrainer`

### 2.2 LoRA — Low-Rank Adaptation

**Core Concept & Mechanics.** Instead of updating all parameters in a high-dimensional weight matrix $W_0 \in \mathbb{R}^{d \times k}$ during training, LoRA freezes $W_0$ and tracks updates through a low-rank decomposition $\Delta W = BA$, where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, and rank $r \ll \min(d, k)$.

1. **Matrix Decomposition:** Setting a low rank (e.g., $r=8$) on a $4096 \times 4096$ matrix reduces trainable parameters from ~16 million down to ~65,000. Because intrinsic fine-tuning dimensionality is low, this gets within 1–2% of full fine-tuning performance using 0.5–2% of the parameters.
2. **Parallel Forward Pass:** Input vectors flow through the frozen base weights and low-rank adapter path simultaneously:

$$
h = W_0 x + \frac{\alpha}{r} BA x
$$

3. **Scaling Factor:** The update path is scaled by a constant factor $\frac{\alpha}{r}$ to stabilize training across rank choices.
4. **Zero-Inference Overhead:** Upon completion, $BA$ can be permanently merged back into $W_0$, eliminating runtime latency.

* **Configs:** `configs/sft_lora.yaml` (`use_lora: true, qlora: false`)
* **Trainer:** `peft.get_peft_model` + `trl.SFTTrainer`

### 2.3 QLoRA — 4-bit Quantized LoRA

**Core Concept & Mechanics.** QLoRA compresses the frozen base model into 4-bit precision while maintaining trainable LoRA adapters in higher-precision 16-bit format.

* **NormalFloat4 (NF4) Quantization:** Maps parameters to 4-bit bins based on standard normal distribution quantiles, which is information-optimal for normally distributed neural network weights.
* **Double Quantization:** Quantizes the quantization scale factors themselves to save additional memory.
* **On-the-Fly Dequantization:** During the forward pass, 4-bit base weights are temporarily dequantized into 16-bit Brain Floating Point (BF16) format to compute activations with incoming vectors, added to the 16-bit LoRA adapter outputs, and immediately discarded from VRAM.
* **Memory Savings:** For a 7B parameter model, QLoRA drops training VRAM requirements from ~80 GB down to ~10 GB with no measurable quality loss.

* **Configs:** `configs/sft_qlora.yaml` (`qlora: true`)

### 2.4 DPO — Direct Preference Optimization

**Core Concept & Mechanics.** DPO aligns models with human preferences using paired comparison data (prompt $x$ with preferred completion $y_w$ and rejected completion $y_l$) without requiring a separate reward model or RL loop.

* **Implicit Reward Modeling:** DPO reparameterizes the RLHF reward as a log-ratio of probability outputs relative to a frozen reference model $\pi_{\text{ref}}$.
* **Direct Optimization:** Both completions pass through the active policy $\pi_\theta$ and frozen reference policy $\pi_{\text{ref}}$ simultaneously to compute log-likelihood ratios under the following objective:

$$
\mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[\log \sigma\!\left(\beta \log\frac{\pi_\theta(y_w \mid x)}{\pi_\text{ref}(y_w \mid x)} - \beta \log\frac{\pi_\theta(y_l \mid x)}{\pi_\text{ref}(y_l \mid x)}\right)\right]
$$

* **Log-Ratio Penalties:** The scaling parameter $\beta$ controls how far the active policy $\pi_\theta$ is allowed to drift from $\pi_{\text{ref}}$, preventing mode collapse or language degradation.

* **Config:** `configs/dpo.yaml`
* **Trainer:** `trl.DPOTrainer` (`loss_type = "sigmoid"`)

### 2.5 ORPO — Odds-Ratio Preference Optimization

**Core Concept & Mechanics.** ORPO merges supervised fine-tuning (SFT) and preference alignment into a single training step directly on a base model, eliminating the need for a frozen reference model and reducing VRAM usage by ~50% compared to DPO.

* **Combined Loss Function:** Adds an odds-ratio penalty on the rejected response directly to the standard SFT negative log-likelihood loss:

$$
\mathcal{L}_{\text{ORPO}} = \mathcal{L}_{\text{SFT}}(y_w) + \lambda \cdot \mathcal{L}_{\text{OR}}
$$

* **Odds-Ratio Penalty:** Measures the odds of generating the preferred completion versus the rejected completion under the current policy, increasing penalties if high probability is assigned to rejected outputs relative to preferred ones.

### 2.6 KTO — Kahneman-Tversky Optimization

**Core Concept & Mechanics.** Inspired by prospect theory in behavioral economics—which posits that humans weigh losses more heavily than gains—KTO optimizes models using pointwise binary feedback (individual responses labeled as "desirable" or "undesirable") rather than paired preferences ($y_w \succ y_l$).

* **Asymmetric Utilities:** Evaluates completions against policy and reference models using asymmetric utility curves:

$$
\mathcal{L}_{\text{KTO}} = \lambda_D \cdot u_D + \lambda_U \cdot u_U
$$

* **Loss Weighting:** $u_D$ uses a concave gain curve for desirable outputs, while $u_U$ applies a steeper loss penalty curve for undesirable outputs. This ensures the model heavily avoids unacceptable outputs.

### 2.7 IPO — Identity Preference Optimization

**Core Concept & Mechanics.** Prevents the "logit collapse" of standard DPO (where the probability gap between preferred and rejected completions is driven infinitely high, causing overconfidence) by replacing DPO's log-sigmoid loss with a regularized squared loss targeting a fixed margin.

* **Enabled by:** Setting `loss_type: "ipo"` in `configs/dpo.yaml`.

### 2.8 CPO — Contrastive Preference Optimization

**Core Concept & Mechanics.** A memory-efficient, reference-free alignment approach combining contrastive loss with a behavior-cloning (BC) regularizer:

$$
\mathcal{L}_{\text{CPO}} = \mathcal{L}_{\text{Contrast}} + \alpha \cdot \mathcal{L}_{\text{BC}}
$$

* **Contrastive Loss:** Penalizes the likelihood gap directly between chosen and rejected generations without keeping a second reference model in VRAM.
* **Behavior Cloning Regularizer:** Anchors the policy to high-quality chosen responses ($\mathcal{L}_{\text{BC}}$), preventing text generation quality drift on small datasets.

### 2.9 Reward Modeling — Bradley-Terry

**Core Concept & Mechanics.** Fits a scalar reward model $r_\theta(x, y)$ that converts a prompt-response pair into a single score value, acting as an automated judge for RL steps (e.g., PPO, GRPO, RLOO).

* **Architecture:** Uses a standard base language model fitted with a single-scalar output head (`AutoModelForSequenceClassification(num_labels=1)`).
* **Bradley-Terry Preference Probability:** Predicts the probability that response $y_w$ is preferred over $y_l$ given prompt $x$:

$$
P(y_w \succ y_l \mid x) = \sigma\!\left(r_\theta(x, y_w) - r_\theta(x, y_l)\right)
$$

* **Objective:** Trained by minimizing $-\log P(y_w \succ y_l)$.

* **Config:** `configs/reward.yaml`
* **Trainer:** `trl.RewardTrainer`

### 2.10 Plackett-Luce Reward Modeling

**Core Concept & Mechanics.** Generalizes Bradley-Terry pairwise preference modeling to evaluate ranked lists of $K$ candidate outputs ($y_1 \succ y_2 \succ \dots \succ y_K$) generated for a single prompt.

* **Iterative Likelihood Factorization:** Evaluates multi-response rankings by computing softmax probabilities over candidate scores sequentially:

$$
P(\text{rank}) = \prod_{k=1}^{K} \frac{\exp r(y_k)}{\sum_{j \ge k} \exp r(y_j)}
$$

* **Efficiency:** Increases sample efficiency by capturing richer human feedback structure per annotation batch.

### 2.11 GRPO — Group Relative Policy Optimization

**Core Concept & Mechanics.** An RL algorithm (popularized by DeepSeek-R1) that eliminates the secondary memory-heavy critic/value model by calculating baselines directly from a group of candidate outputs sampled for the same prompt.

* **Group Sampling:** For each prompt $x$, samples $G$ candidate completions.
* **Reward Evaluation:** Scores completions using target reward functions (e.g., checking code syntax, math correctness, or output formatting).
* **Normalized Advantage Calculation:** Calculates the advantage $\hat{A}_i$ for completion $i$ using its z-score relative to the group mean and standard deviation:

$$
\hat{A}_i = \frac{r_i - \text{mean}(r_{1..G})}{\text{std}(r_{1..G})}
$$

* **Config:** `configs/grpo.yaml` (`reward_funcs: [length, format, no_repetition]`)
* **Trainer:** `trl.GRPOTrainer`
* **Reward Functions:** Defined in `src/training/reward_funcs.py`

> **Important TRL 0.14.0 Gotchas.** `GRPOConfig` does not accept `top_p`, `max_length` (use `max_completion_length`), or `log_completions` in this version. `reward_funcs` is a required positional argument.

### 2.12 RLOO — REINFORCE Leave-One-Out

**Core Concept & Mechanics.** Uses the same group-sampling structure ($G$ completions per prompt) as GRPO to eliminate the critic network, but defines the baseline for completion $i$ as the leave-one-out mean of the remaining $G-1$ completions:

$$
\hat{A}_i = r_i - \frac{1}{G-1}\sum_{j \ne i} r_j
$$

This baseline yields an unbiased advantage estimate with lower variance than standard REINFORCE methods at a lower computational cost than PPO.

### 2.13 Online DPO

**Core Concept & Mechanics.** Overcomes off-policy dataset drift by forcing the policy model to generate its own completions dynamically during training.

* **Rollout & Real-time Scoring:** The active model generates candidate outputs, which are scored on the fly by an automated reward model or LLM-as-a-judge to establish winning ($y_w$) and losing ($y_l$) pairs dynamically.
* **Immediate Update:** The freshly generated preference pairs are immediately processed using the standard DPO loss function.

### 2.14 Nash-MD — Nash Mirror Descent

**Core Concept & Mechanics.** Frames alignment as a two-player zero-sum game between the evolving model policy and a reference policy. By updating parameters using mirror descent bounded by KL divergence constraints, the model converges to a Nash equilibrium in policy space, making performance robust against circular or contradictory human preference data.

### 2.15 XPO — Exploratory Preference Optimization

**Core Concept & Mechanics.** Enhances sample efficiency on limited preference datasets by adding an exploration reward bonus to the standard DPO objective. This bonus encourages the policy to sample responses in uncertain regions of the reward landscape, avoiding premature convergence on local optima.

### 2.16 Knowledge Distillation — GKD (Generative Knowledge Distillation)

**Core Concept & Mechanics.** Transfers knowledge from a large teacher model to a smaller student model by matching full next-token output probability distributions across the entire vocabulary rather than relying only on hard argmax targets.

* **On-Policy Sampling:** The student model generates output sequences on training prompts.
* **Distribution Matching:** The student minimizes the KL divergence between its output logits and the teacher's soft probability distribution on its own generated outputs, teaching the student how to recover from its own generation errors.

### 2.17 MiniLLM

**Core Concept & Mechanics.** Replaces standard Forward KL divergence with Reverse KL divergence:

$$
\mathcal{L}_{\text{MiniLLM}} = D_{KL}(\pi_\theta \,\Vert{}\, \pi_T)
$$

* **Mode-Focusing:** Forward KL forces a low-capacity student to spread probability mass across all modes of a complex teacher distribution (often resulting in ungrammatical or incoherent generations). Reverse KL penalizes the student for outputting tokens considered improbable by the teacher, focusing the student on mastering high-confidence modes it can accurately represent.
* **Gradient Estimation:** Employs policy-gradient expectation updates to calculate Reverse KL gradients efficiently over long sequences.

### 2.18 Meta-Learning — MAML, Reptile, Meta-ICL

Configures base parameters so the model can adapt rapidly to new target tasks given very few examples.

* **MAML (Model-Agnostic Meta-Learning):** Bi-level optimization loop. The model takes gradient steps on a small support set for a task (inner loop), evaluates performance on a query set (outer loop), and backpropagates through the inner loop updates to optimize initial weight settings.
* **Reptile:** A first-order approximation of MAML that avoids backpropagating through inner-loop gradient paths. It trains on a task for standard steps, then pulls base weights toward post-adaptation weights: $\theta \leftarrow \theta + \epsilon(\theta' - \theta)$.
* **Meta-ICL (Meta In-Context Learning):** Fine-tunes the language model on structured episodes ([example 1] [example 2] [example 3] [query] -> answer) to train internal attention mechanisms to process and leverage in-context demonstrations at inference time.

---

# 3. Evaluation

Evaluation in this repository happens at two distinct stages of the pipeline, and each stage tracks a different family of metrics. The rest of this section documents both stages and details the foundational mechanics behind every evaluated method and metric.

## 3.1 Training Metrics (per method)

Training-time metrics are recorded automatically during training to monitor whether the model is learning smoothly. They track optimization health, policy stability, and numerical convergence. Because different training algorithms optimize fundamentally different objectives, these metrics are **not** comparable across methods—their sole purpose is to verify that a specific run converged cleanly.

### 3.1.1 SFT (and SFT+LoRA, SFT+QLoRA)

**What Supervised Fine-Tuning Is.** SFT adapts a pre-trained base model to follow instructions or adopt a target output style. It trains the model to predict the next token in reference answers given a prompt. It teaches the model how to structure responses like a helpful assistant, but it does not teach it *which* response is preferred when multiple valid answers exist.

**Logged Metrics:**

* `loss` **(Cross-Entropy Loss):** Measures how surprised the model is by the correct reference tokens. The model outputs a probability distribution over the vocabulary for each target word; this metric calculates the negative log probability assigned to the actual target word. Lower values mean the model is placing higher probability on the correct answer text.
* `eval_loss` **(Validation Loss):** The exact same cross-entropy calculation performed on a held-out validation dataset that the model never trains on. It tracks generalizability.

* `learning_rate` **:** Tracks the current step size used by the optimizer. It rises during the initial warmup phase to prevent early gradient shock, then gradually decays according to a cosine schedule to help the model settle into a stable local minimum.

* `grad_norm` **(Gradient L2 Norm):** Measures the overall magnitude of parameter updates across the network before gradient clipping is applied. Spikes in gradient norm indicate sudden instability, exploding gradients, or corrupted batch data.

* `epoch` / `step` **:** Step-level and pass-level progress counters tracking overall training duration.

### 3.1.2 DPO (and its variants: IPO, ORPO, CPO, KTO)

**What DPO-Family Methods Are.** Direct Preference Optimization and its variants align models with human preferences using comparison data (a prompt alongside a winning response and a losing response). Instead of training a complex, secondary reinforcement learning reward model, DPO mathematically reparameterizes the preference problem, allowing the policy network itself to act as its own implicit reward model relative to a frozen reference model.

**Logged Metrics:**

* `rewards/chosen`: The implicit numeric score assigned to the preferred answer. It is computed from the log-ratio of the probability assigned to the chosen answer by the active model versus the probability assigned to it by the frozen reference model, scaled by a hyperparameter.

* `rewards/rejected`: The implicit numeric score assigned to the dispreferred answer. Computed similarly to the chosen reward, it tracks how much the active model favors or penalizes the rejected response relative to the reference base model.

* `rewards/margins`: The mathematical difference between the chosen reward and the rejected reward. This is the primary value the loss function maximizes; a steadily growing margin shows the model is successfully learning to distinguish good answers from bad ones.

* `rewards/accuracies`: The percentage of batch samples where the active model assigns a strictly higher implicit reward to the preferred completion than to the rejected completion.

* `logps/chosen`: The raw log-probability the model assigns to generating the exact sequence of tokens in the preferred completion. Tracking this ensures the model maintains overall text generation fluency and does not crash its probability distribution to exploit preference scores.

* `logps/rejected`: The raw log-probability assigned to the sequence of tokens in the rejected completion.

* `objective/kl` **(Kullback-Leibler Divergence):** Measures how much the active policy model's probability distribution has drifted away from the original frozen reference model. Bounded KL divergence prevents the model from "reward hacking"—generating degenerate, ungrammatical, or repetitive text just to score high preference margins.

* `loss`: The overall scalar training loss driving optimization.

**Variant-Specific Diagnostics:**

* **KTO (Kahneman-Tversky Optimization):** Based on behavioral economics (Prospect Theory), KTO uses binary labels ("thumbs up" / "thumbs down") rather than paired A/B choices. Metrics are split by label: `rewards/desirable`, `rewards/undesirable`, `rewards/margins`, and `rewards/accuracies`.

* **ORPO (Odds-Ratio Preference Optimization):** Combines instruction fine-tuning and preference learning into a single step without needing a reference model. Logs two distinct loss terms: `loss/sft_loss` (which anchors general language modeling fluency) and `loss/odds_ratio_loss` (which penalizes the odds of generating rejected outputs).

* **CPO (Contrastive Preference Optimization):** A memory-efficient, reference-free method that uses a contrastive loss paired with a behavior-cloning penalty. It logs `loss/contrastive` (which maximizes the gap between chosen and rejected outputs) and `loss/bc` (which forces the model to maintain high generation quality on the chosen completion).

### 3.1.3 Reward Modeling (Bradley-Terry, Plackett-Luce)

**What Reward Models Are.** Reward modeling trains a dedicated evaluator model to assign a single scalar score to any prompt-response pair, reflecting human approval. Bradley-Terry reward models evaluate pair comparisons (*A > B*), while Plackett-Luce models generalize this mechanism to evaluate and rank lists of three or more completions simultaneously. The resulting model acts as an automated "judge" to guide downstream reinforcement learning algorithms like GRPO or RLOO.

**Logged Metrics:**

* `accuracy`: The proportion of evaluated test pairs where the trained reward model successfully outputs a higher numerical score for the human-preferred response than for the rejected response.

* `rewards/margins`: The average numerical score gap between chosen and rejected responses across the current training batch.

* `rewards/chosen` / `rewards/rejected`: The raw scalar numbers output by the model's final score head for both classes.

* `loss`: The negative log-likelihood of correctly predicting the preference ordering.

* `mean_reward`: The average output score across all inputs. Used to verify that scores remain centered near zero rather than drifting to extreme positive or negative values.

### 3.1.4 GRPO (Group Relative Policy Optimization)

**What GRPO Is.** GRPO is an online reinforcement learning algorithm (popularized by DeepSeek-R1) designed to eliminate the need for a memory-heavy critic network. For every prompt, the model generates a group of distinct candidate completions. It scores each completion using a Python reward function or a reward model, computes the average score of the group, and calculates each completion's advantage as its relative z-score above or below that group average.

**Logged Metrics:**

* `reward/mean`: The average score earned by all sampled completions across all prompts in the batch.

* `reward/std`: The standard deviation (spread) of scores within generated groups. A standard deviation near zero indicates that all completions in a group are receiving identical scores, which stops parameter updates completely.

* `kl`: The relative entropy shift measuring how far the policy model has drifted from the initial reference policy during RL exploration.

* `completions/mean_length`: The average length (in tokens) of the model's generated outputs. Monitoring this detects "verbosity hacking," where RL models learn that writing longer answers tricks the reward function into giving higher scores.

* `completions/max_length`: The length of the single longest generation in the batch, monitored to ensure outputs do not repeatedly hit the hard generation ceiling.

* `frac_reward_zero_std`: The proportion of prompts in a batch where all generated outputs received the exact same reward score. A high value means the reward function is either too easy (everything passes) or too difficult (everything fails).

* `loss`: The clipped policy-gradient loss driving policy parameter updates.

* `learning_rate`: The active step size following the decay schedule.

### 3.1.5 RLOO, Online DPO, Nash-MD, XPO (Online RL Family)

**What Online RL Methods Are.** Unlike static preference methods that train on pre-collected offline datasets, Online RL algorithms force the model to generate fresh completions during training. These rollouts are scored dynamically by a reward function or judge model to provide immediate feedback.

* **RLOO (REINFORCE Leave-One-Out):** Generates a group of completions and scores each output against a baseline formed by averaging all other completions in that same group.

* **Online DPO:** Dynamically generates pairs of completions, scores them on the fly, and applies the DPO loss immediately to prevent off-policy drift.

* **Nash-MD (Nash Mirror Descent):** Treats preference alignment as a two-player zero-sum game between the active policy and a reference policy, using mirror descent updates to reach a stable Nash equilibrium that resists contradictory human preferences.

* **XPO (Exploratory Preference Optimization):** Adds an exploration bonus reward to target under-explored prompt spaces where preference estimates remain uncertain.

**Logged Metrics:**

* `objective/rlhf_reward`: The total reward score after subtracting the KL divergence penalty. This represents the primary indicator of successful preference alignment.

* `objective/scores`: The raw score generated by the reward evaluator before applying penalties.

* `objective/scores_margin`: The average difference between high-scoring and low-scoring generations generated in the same rollout batch (specifically for Online DPO).

* `objective/kl`: The divergence between the active policy and the initial base policy.

* `objective/entropy`: A measure of randomness in the model's output probability distribution. If entropy collapses to near zero, the model has become completely deterministic and can no longer explore diverse answers.

* `rewards/accuracies`: How often the policy's internal preferred completion matches the external judge's ranking.

* `rewards/margins`: The score gap between candidate generations.

* `policy/loss`: The surrogate policy-gradient loss driving network updates.

* `val/num_eos_tokens`: The proportion of generated outputs that correctly ended with an End-Of-Sequence token. A drop indicates that the model is generating infinite loops or truncated text.

### 3.1.6 Knowledge Distillation (GKD, MiniLLM)

**What Knowledge Distillation Is.** Knowledge distillation transfers reasoning capacity and generation quality from a large, highly capable "teacher" model into a smaller, faster "student" model.

* **GKD (Generative Knowledge Distillation):** Forces the student model to match the teacher's soft probability distribution across the entire vocabulary on sequences generated by the student itself.

* **MiniLLM:** Uses Reverse KL divergence instead of standard Forward KL divergence. Standard distillation forces a small student to cover every complex output mode of a teacher (often causing ungrammatical output). Reverse KL forces the student to focus exclusively on mastering the subset of high-confidence modes it has the parameter capacity to express accurately.

**Logged Metrics:**

* `loss`: The primary combined distillation loss.

* `loss/ce`: Hard cross-entropy loss comparing student predictions directly against ground-truth dataset labels.

* `loss/kd`: Soft divergence loss measuring how closely the student's probability distribution matches the teacher's probability distribution.

* `forward_kl` **(MiniLLM):** Tracks standard forward divergence as an informational baseline metric.

* `reverse_kl` **(MiniLLM):** Tracks reverse divergence—the core optimization target that focuses the student model on high-confidence output modes.

* `accuracy`: The proportion of tokens where the student model's top predicted word matches the teacher model's top predicted word.

### 3.1.7 Meta-Learning (MAML, Reptile, Meta-ICL)

**What Meta-Learning Methods Are.** Meta-learning trains models *how to adapt*. Instead of optimizing parameters for a single fixed task, meta-learning conditions the model so it can adapt to entirely new tasks using only a few examples.

* **MAML (Model-Agnostic Meta-Learning):** Uses a bi-level optimization loop. It takes trial gradient steps on a small "support set" for a task (inner loop), evaluates the resulting performance on a "query set" (outer loop), and optimizes the starting weights so that a few gradient steps on any future task yield high accuracy.

* **Reptile:** A computationally efficient alternative to MAML that executes standard training steps on a task and then pulls the original base weights toward the newly adapted weights.

* **Meta-ICL (Meta In-Context Learning):** Fine-tunes the model on structured multi-task task episodes ([Example 1] [Example 2] [Query] -> Target), conditioning its internal attention mechanisms to extract maximum signal from in-context examples provided at runtime.

**Logged Metrics:**

* `meta_train_loss`: The performance error on task query sets after applying inner-loop adaptation steps.

* `meta_val_loss`: The adaptation performance error measured on held-out tasks the model never encountered during training.

* `meta_accuracy`: The percentage of correct query predictions produced after completing inner-loop task adaptation.

* `inner_loss`: Diagnostic error measured on the small support set during initial inner-loop adaptation steps.

* `weight_distance`: The total magnitude of change between the base model weights and the post-adaptation weights. A value near zero means the model failed to adapt its parameters during the task update step.

## 3.2 Evaluation Metrics

Once training completes, `scripts/evaluate.py` (or `notebooks/02_evaluation_comparison.ipynb`) loads every saved adapter checkpoint and benchmarks it against the held-out test set using three standard text-generation metrics and diagnostic length trackers.

### 3.2.1 BLEU (Bilingual Evaluation Understudy)

**What It Is.** BLEU is a precision-focused text similarity metric originally designed for machine translation. It measures how many word sequences (single words, two-word pairs, three-word triplets, and four-word phrases) in the model's generated text appear in the human reference answer.

**How It Works Conceptually.**

* **Phrase Matching:** It checks $1$-gram, $2$-gram, $3$-gram, and $4$-gram phrase overlaps between prediction and reference.

* **Clipping:** To prevent cheating (e.g., repeating the same correct word over and over), word match counts are capped at the maximum number of times that word actually appears in the reference.

* **Brevity Penalty:** It applies an automatic penalty multiplier if the generated output is significantly shorter than the reference answer, preventing the model from achieving high precision by outputting only a single highly confident word.

* **Geometric Combination:** Precision scores across all phrase lengths are combined into a single geometric mean.

**Range.** Scale of $0.0$ to $1.0$ (after dividing the raw score by 100).

**Interpretation.** BLEU evaluates output **precision**—how much of what the model wrote matches the reference text exactly. Because it relies heavily on exact phrasing, it favors conservative, literal answers and heavily penalizes valid rephrasings that use alternative vocabulary.

### 3.2.2 ROUGE-L (Recall-Oriented Understudy for Gisting Evaluation — Longest Common Subsequence)

**What It Is.** ROUGE-L is a recall-leaning text evaluation metric designed for summarization and open-ended text generation. It measures the length of the longest sequence of words that appear in both the generated text and the reference text in the exact same relative order, even if they are separated by other words.

**How It Works Conceptually.**

* **Sequence Order Alignment:** It identifies the longest common chain of words shared by both texts without requiring contiguous matching. For example, "The cat sat on mat" and "The small cat sat down on the mat" share a long ordered chain despite inserted adjectives.

* **Precision and Recall Calculation:** It computes two numbers: Sequence Precision (the length of the matching chain divided by total generated length) and Sequence Recall (the length of the matching chain divided by total reference length).

* **Harmonic Mean (F-Measure):** It balances precision and recall into a single harmonic F-measure score.

**Range.** Scale of $0.0$ to $1.0$.

**Interpretation.** ROUGE-L evaluates **content coverage and sentence structure**. It measures how completely the model captured the core ideas of the reference text in logical order. It is far less brittle than BLEU regarding exact word choice because matching words do not need to sit directly next to one another.

### 3.2.3 Exact Match (EM)

**What It Is.** Exact Match is a strict binary accuracy metric that checks whether the model's generated text is identical to the ground-truth reference text after basic string standardization.

**How It Works Conceptually.**

* **Text Normalization:** Lowercases all characters, strips leading and trailing space, and collapses multiple consecutive spaces or newline characters into single spaces.

* **Binary Comparison:** Returns a score of $1.0$ if the normalized prediction string perfectly matches the normalized reference string, and $0.0$ otherwise.

* **Dataset Averaging:** Calculates the overall fraction of identical predictions across the evaluation set.

**Range.** Scale of $0.0$ to $1.0$.

**Interpretation.** The strictest generation metric available. It is suited only for closed-domain tasks with single canonical answers (such as multiple-choice classification labels, extraction tasks, or single-number math outputs). On open-ended technical generation tasks (such as writing code or answering StackOverflow questions), EM scores are expected to be **0.0**, as there are infinitely many valid ways to write code or prose.

### 3.2.4 Length Statistics (Diagnostic)

Along with text quality scores, the post-hoc evaluation script logs diagnostic structural statistics for every evaluated model output:

* `mean_length`: The average number of whitespace-separated words produced in generated responses.

* `max_length`: The token length of the single longest response produced across the test set.

* `n`: The total count of test samples evaluated.

**Failure Modes Revealed:**

* **Generation Truncation:** If `max_length` matches `max_new_tokens` continuously across many test samples, the model is failing to emit natural End-Of-Sequence (EOS) tokens, causing answers to cut off mid-sentence.

* **Verbosity Hacking:** If a fine-tuned model exhibits a dramatically larger `mean_length` than the human references, it has likely exploited reward setups that unintentionally favor long, wordy answers over concise ones.

---

## 4. Repository Structure

```
a-survey-of-LLMs-fine-tuning-approaches/
│
├── configs/                       # ★ The single source of truth for every run
│   ├── base.yaml                  # Shared defaults (model, seed, LoRA params)
│   ├── sft_lora.yaml              # One file per experiment
│   ├── sft_qlora.yaml
│   ├── dpo.yaml
│   ├── reward.yaml
│   ├── grpo.yaml
│   └── eval.yaml                  # Evaluation config (models, metrics, output path)
│
├── src/
│   ├── data/
│   │   ├── preprocess.py          # load_stackoverflow, split_qa, make_*_dataframe
│   │   └── dataset.py             # make_train_val_datasets (index-safe splitting)
│   │
│   ├── models/
│   │   ├── loaders.py             # load_tokenizer, load_causal_lm, load_reward_model,
│   │   │                          #   load_model_for_inference (handles PEFT adapters)
│   │   └── peft.py                # apply_lora wrapper (torchao-safe)
│   │
│   ├── training/
│   │   ├── sft.py                 # train_sft(cfg, train_ds, eval_ds)
│   │   ├── dpo.py                 # train_dpo(cfg, train_ds, eval_ds)
│   │   ├── reward.py              # train_reward(cfg, train_ds, eval_ds)   ← trainer
│   │   ├── reward_funcs.py        # reward functions for GRPO                ← callables
│   │   └── grpo.py                # train_grpo(cfg, train_ds, eval_ds)
│   │
│   ├── eval/
│   │   ├── generate.py            # deterministic batch generation
│   │   ├── metrics.py             # BLEU / ROUGE-L / Exact Match
│   │   └── evaluate.py            # evaluate_models(cfg) — driven by configs/eval.yaml
│   │
│   └── utils/
│       ├── version.py             # version-detection flags (TRL, transformers)
│       ├── config.py              # TRL config builders (kwargs-safe)
│       ├── config_loader.py       # YAML loader with `defaults:` inheritance
│       └── seed.py                # set_seed
│
├── scripts/                       # CLI entry points
│   ├── train_sft.py               # python scripts/train_sft.py    --config configs/sft_lora.yaml
│   ├── train_dpo.py               # python scripts/train_dpo.py    --config configs/dpo.yaml
│   ├── train_reward.py            # python scripts/train_reward.py --config configs/reward.yaml
│   ├── train_grpo.py              # python scripts/train_grpo.py   --config configs/grpo.yaml
│   └── evaluate.py                # python scripts/evaluate.py     --config configs/eval.yaml
│
├── notebooks/
│   ├── 01_fine_tuning_comparison.ipynb   # End-to-end: data -> train all methods
│   └── 02_evaluation_comparison.ipynb    # BLEU / ROUGE-L / EM benchmark + plots
│
├── data/
│   └── stackoverflow/
│       └── stacksample/           # Populated by the Kaggle download step
│           ├── Questions.csv
│           └── Answers.csv
│
├── outputs/                       # All trained adapters land here
│   ├── sft-lora/
│   ├── sft-qlora/
│   ├── dpo/
│   ├── reward/
│   ├── grpo/
│   ├── evaluation_metrics.csv
│   └── evaluation_comparison.png
│
├── requirements.txt
└── README.md                      # ← you are here
```

---

## 5. Environment Setup (Step-by-Step)

### 5.1 Clone the repository

```bash
git clone https://github.com/<your-username>/a-survey-of-LLMs-fine-tuning-approaches.git
cd a-survey-of-LLMs-fine-tuning-approaches
```

### 5.2 Create a Python environment

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

### 5.3 Install pinned dependencies

The repository was developed and validated against the following pinned stack.
Do **not** upgrade TRL without updating `src/utils/version.py` — TRL renames
config fields between versions.One shot installation:

```bash
pip install -r requirements.txt
```

> **Note on torchao.** Some environments ship a broken `torchao` build whose
> compiled extensions fail to load against PyTorch 2.10/2.11. The repo has a
> built-in guard (`src/utils/version.py` + the notebook's `_TorchaoBlocker`)
> that hides `torchao` from the import system. Optionally uninstall it:
> ```bash
> pip uninstall -y torchao
> ```

### 5.4 Set up the Kaggle API credentials

The StackSample dataset is downloaded from Kaggle. You must supply your own
`kaggle.json` API token.

1. Log in to Kaggle -> **Account -> Create New API Token**. This downloads
   `kaggle.json` containing your username and API key.
2. Move it to the standard location:

   ```bash
  pip install kaggle
  mkdir -p ~/.kaggle
  mv /your_download_path/kaggle.json ~/.kaggle/
  chmod 600 ~/.kaggle/kaggle.json
   ```

   On Windows, the file goes to `%USERPROFILE%\.kaggle\kaggle.json`.

3. Verify:

   ```bash
   kaggle datasets list -s stacksample
   ```

   You should see `stackoverflow/stacksample` in the output.

### 5.5 Hugging Face access (optional)

Some gated models (e.g., Gemma) require a Hugging Face token. Export it:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

---

## 6. Dataset Preparation

The repository uses the **StackSample** dataset (StackOverflow Questions +
Answers, ~500K rows, CC-BY-SA 3.0). Only the top 200 highest-scoring
questions are kept per run to keep training tractable.

### 6.1 Download

From the repository root:

```bash
# Download and extract into data/stackoverflow/stacksample/
python -m src.data.download   # (or run the "Download Dataset" cell in notebook 01)
```

Or manually:

```bash
import kaggle
kaggle.api.authenticate()
kaggle.api.dataset_download_files("stackoverflow/stacksample", path="/a-survey-of-LLMs-fine-tuning-approaches/data",unzip=True)
```

After this you should have:

```
data/stackoverflow/stacksample/
├── Questions.csv
└── Answers.csv
```

### 6.2 Raw schema

| File | Columns used |
|---|---|
| `Questions.csv` | `Id`, `Title`, `Body`, `Score` |
| `Answers.csv`   | `Id`, `ParentId`, `Body`, `Score` |

### 6.3 Cleaning pipeline (in `src/data/preprocess.py`)

1. **HTML stripping** — `BeautifulSoup` removes `<p>`, `<code>`, etc.
2. **Quality filter** — keep only rows with `Score > 5`.
3. **Join** — `Answers.ParentId == Questions.Id` (inner join).
4. **Task-specific framing:**
   * SFT -> `Question: …\nAnswer: …`
   * DPO -> `{prompt, chosen, rejected}` by max/min score per question
   * Reward -> all-pairs tournament per question
   * GRPO -> `{prompt, reference}` only
5. **Split** — 80/10/10 by question ID (train/val/test).

### 6.4 Why splits are done on the DataFrame, not on the `Dataset`

`datasets.Dataset.from_pandas(..., preserve_index=False)` **discards the
original index**, so the pattern

```python
train = Dataset.from_pandas(df.sample(frac=0.8))
val   = Dataset.from_pandas(df.drop(train.to_pandas().index))   # ← KeyError
```

is a classic bug. The repository instead uses
`src/data/dataset.py::make_train_val_datasets(df, frac=0.8, seed=42)`,
which splits the DataFrame first and converts to `Dataset` afterwards.

---

## 7. Configuration System

Every hyperparameter lives in **YAML**, never in Python. The loader
(`src/utils/config_loader.py`) supports **`defaults:`** inheritance so
method-specific files only override what differs from `base.yaml`.

### 7.1 Example — `configs/base.yaml`

```yaml
seed: 42
report_to: "none"
fp16: false
bf16: false
gradient_checkpointing: true

model_name: "google/gemma-3-270m"
data_dir: "data/stackoverflow/stacksample"

lora:
  r: 8
  alpha: 32
  dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
```

### 7.2 Example — `configs/sft_lora.yaml`

```yaml
defaults: [base]

method: sft
output_dir: "outputs/sft-lora"
use_lora: true
qlora: false

max_seq_length: 256
dataset_text_field: "text"
packing: false

epochs: 1
batch_size: 4
grad_accum: 2
lr: 2.0e-4
logging_steps: 5
eval_strategy: "epoch"
save_strategy: "epoch"
```

### 7.3 Loading programmatically

```python
from src.utils.config_loader import load_config

cfg = load_config("configs/grpo.yaml")
print(cfg["model_name"], cfg["num_generations"], cfg["reward_funcs"])
# -> google/gemma-3-270m 4 ['length', 'format', 'no_repetition']
```

All `defaults:` chains are resolved recursively and deep-merged.

---

## 8. Running Experiments

### 8.1 CLI (recommended for reproducibility)

```bash
# Supervised Fine-Tuning
python scripts/train_sft.py    --config configs/sft_lora.yaml
python scripts/train_sft.py    --config configs/sft_qlora.yaml

# Preference optimization
python scripts/train_dpo.py    --config configs/dpo.yaml

# Reward modeling
python scripts/train_reward.py --config configs/reward.yaml

# Online RL
python scripts/train_grpo.py   --config configs/grpo.yaml
```

### 8.2 From a notebook

Open `notebooks/01_fine_tuning_comparison.ipynb`. It walks through:

1. Environment check + quiet-mode setup
2. Data loading (Kaggle download + split)
3. All four trainings, each driven by its YAML
4. A final summary table of where each artefact landed

Every training call has the same shape:

```python
from src.utils.config_loader import load_config
from src.training.sft import train_sft

cfg = load_config("configs/sft_lora.yaml")
train_sft(cfg, train_ds, val_ds)
```

### 8.3 Output artefacts

Every run writes to `output_dir` (from the YAML):

```
outputs/sft-lora/
├── adapter_config.json         # PEFT adapter metadata
├── adapter_model.safetensors   # the trained adapter (~50 MB)
├── tokenizer.json
├── tokenizer_config.json
└── special_tokens_map.json
```

Full models (Reward, etc.) additionally include `model.safetensors`.

---

## 9. Citation & References

If you use this repository for academic work, please cite the methods it
implements. Selected references:

```bibtex
@article{hu2021lora,
  title={LoRA: Low-Rank Adaptation of Large Language Models},
  author={Hu, Edward J. and Shen, Yelong and Wallis, Phillip and others},
  journal={arXiv:2106.09685}, year={2021}}

@article{dettmers2023qlora,
  title={QLoRA: Efficient Finetuning of Quantized LLMs},
  author={Dettmers, Tim and Pagnoni, Artidoro and Holtzman, Ari and Zettlemoyer, Luke},
  journal={arXiv:2305.14314}, year={2023}}

@article{rafailov2023dpo,
  title={Direct Preference Optimization: Your Language Model is Secretly a Reward Model},
  author={Rafailov, Rafael and Sharma, Archit and Mitchell, Eric and others},
  journal={arXiv:2305.18290}, year={2023}}

@article{shao2024deepseekmath,
  title={DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models},
  author={Shao, Zhihong and Wang, Peiyi and Zhu, Qihao and others},
  journal={arXiv:2402.03300}, year={2024}}

@article{ethayarajh2024kto,
  title={KTO: Model Alignment as Prospect Theoretic Optimization},
  author={Ethayarajh, Kawin and Xu, Winnie and Muennighoff, Niklas and others},
  journal={arXiv:2402.01306}, year={2024}}

@article{hong2024orpo,
  title={ORPO: Monolithic Preference Optimization without Reference Model},
  author={Hong, Jiwoo and Lee, Noah and Thorne, James},
  journal={arXiv:2403.07691}, year={2024}}

@article{gu2024minillm,
  title={MiniLLM: Knowledge Distillation of Large Language Models},
  author={Gu, Yuxian and Dong, Li and Wei, Furu and Huang, Minlie},
  journal={arXiv:2306.08543}, year={2024}}

@article{agarwal2024gkd,
  title={On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes},
  author={Agarwal, Rishabh and Vieillard, Nino and Zhou, Yongchao and others},
  journal={arXiv:2306.13649}, year={2024}}

@article{finn2017maml,
  title={Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks},
  author={Finn, Chelsea and Abbeel, Pieter and Levine, Sergey},
  journal={arXiv:1703.03400}, year={2017}}

@article{nichol2018reptile,
  title={On First-Order Meta-Learning Algorithms},
  author={Nichol, Alexander and Achiam, Joshua and Schulman, John},
  journal={arXiv:1803.02999}, year={2018}}

@article{wu2024loreft,
  title={ReFT: Representation Finetuning for Language Models},
  author={Wu, Zhengxuan and Arora, Aryaman and Wang, Zheng and others},
  journal={arXiv:2404.03592}, year={2024}}

@article{liu2024dora,
  title={DoRA: Weight-Decomposed Low-Rank Adaptation},
  author={Liu, Shih-Yang and Wang, Chien-Yi and Yin, Hongxu and others},
  journal={arXiv:2402.09353}, year={2024}}

@article{kopiczko2024vera,
  title={VeRA: Vector-based Random Matrix Adaptation},
  author={Kopiczko, Dawid J. and Blankevoort, Tijmen and Asano, Yuki M.},
  journal={arXiv:2310.11454}, year={2024}}

@article{zhang2023adalora,
  title={AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning},
  author={Zhang, Qingru and Chen, Minshuo and Bukharin, Alexander and others},
  journal={arXiv:2303.10512}, year={2023}}
```

A full bibliography (60+ entries) is maintained in the research notebook.

---

## License & Acknowledgements

- Code: MIT License.
- Dataset: StackSample is distributed under **CC-BY-SA 3.0**; respect its
  attribution requirements.
- Base model: `google/gemma-3-270m` is distributed under Google's Gemma
  Terms of Use.
- Built on the shoulders of the `transformers`, `trl`, `peft`, `datasets`,
  and `accelerate` teams at Hugging Face.

---

*Last updated: 2026 — for issues, please open a GitHub Issue or submit a PR.*
```
