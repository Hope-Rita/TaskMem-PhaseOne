# TaskMem-Phase One

RL training pipeline for **episodic memory generation**, built on top of
[`volcengine/verl`](https://github.com/volcengine/verl) (forked at
`33eb86f54f75ffc5825bfd54cee71b57b6b8ae13`). The base policy is
**Qwen3-VL-30B-A3B-Thinking**; reward judging is done via Azure OpenAI
(GPT-4o) + Gemini 2.5 Flash through the M3 reward manager.

---

## 1. Install

```bash
git clone <repo-url>
cd TaskMem-PhaseOne
bash bootstrap.sh
```

`bootstrap.sh` installs PyTorch 2.8 / CUDA 12.8, vLLM 0.11, Megatron-LM
core\_v0.13.1, transformer-engine 2.8, flash-attn 2.7.4, and the M3 reward
manager's Python dependencies (`json_repair`, `openai`, etc.).

### Azure OpenAI / Gemini judge config

The M3 reward manager calls an external judge through Azure OpenAI. Copy the
template and fill in your own endpoints / keys (the real file is
gitignored):

```bash
cp configs/api_config.example.json configs/api_config.json
# then edit configs/api_config.json
```

Each model name can be either a single config object or a list of objects
to round-robin across (useful for rate-limit avoidance):

```json
{
  "gpt-4o": { "azure_endpoint": "...", "api_version": "...", "api_key": "..." },
  "gemini-2.5-flash": [
    { "azure_endpoint": "...", "api_version": "...", "api_key": "..." },
    { "azure_endpoint": "...", "api_version": "...", "api_key": "..." }
  ]
}
```

If you store the config elsewhere, point at it via `M3_API_CONFIG`:

```bash
export M3_API_CONFIG=/path/to/your/api_config.json
```

---

## 2. Data format

`VLDataset` (see `verl/utils/dataset/vl_dataset.py`) reads **JSONL** —
one JSON object per line. The required fields depend on `type`:

| Field | Type | Required for | Description |
| --- | --- | --- | --- |
| `id` | str | all | Sample id; format `<video_id>*<key>` (split on `*`) |
| `type` | str | all | One of `"episodic_on_policy"`, `"semantic"` |
| `step` | int | episodic / semantic | Curriculum step. When `> 0` the loader pulls previously generated memories from `<default_local_dir>/history/<video_id>.json` |
| `video_idx` | int | episodic / semantic | Current segment index inside the video |
| `episodic_folder` | str | episodic / semantic | Directory containing `<video_id>/{video_id}_map.json` (face-id mapping) and `episodic_<memory_tag>.json` |
| `memory_tag` | str | semantic only | Selector for `episodic_<memory_tag>.json` |
| `input` | list | all | OpenAI-style content blocks (`{"type": "text"\|"video"\|"image", ...}`); the last text block must contain `[Description of the preceding part]:` so the reward manager can extract preceding context |

The dataset rewrites face ids (`<face_n>`) using the per-frame map files
so that ids remain locally consistent across history segments.

---

## 3. Train

The repository ships `run.sh` with the Megatron / vLLM / offload defaults
already baked in. Supply the user-specific overrides (model path, data
paths, output dir, reward hyperparameters) on the command line:

```bash
bash run.sh \
    data.train_batch_size=32 \
    actor_rollout_ref.rollout.n=8 \
    data.train_files=/path/to/train.jsonl \
    data.val_files=/path/to/val.jsonl \
    actor_rollout_ref.model.path=/path/to/Qwen3-VL-30B-A3B-Thinking \
    trainer.default_local_dir=/path/to/output/run_name \
    trainer.experiment_name=run_name \
    trainer.save_freq=10 \
    reward_model.reward_kwargs.memory_length_threshold=5 \
    reward_model.reward_kwargs.think_penalty=-1.0 \
    reward_model.reward_kwargs.think_length_threshold=1200 \
    reward_model.reward_kwargs.correct_reward=0.5 \
    reward_model.reward_kwargs.wrong_reward=-0.5 \
    reward_model.reward_kwargs.format_penalty=-1.5 \
    reward_model.reward_kwargs.memory_token_threshold=150 \
    reward_model.reward_kwargs.usefulness_scale=0.1 \
    actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
    algorithm.adv_estimator=grpo \
    algorithm.mode=optimize_threshold \
    data.shuffle=False \
    data.dataloader_num_workers=0
```

---

## 4. Outputs

All artifacts go under `trainer.default_local_dir`:

```
<default_local_dir>/
├── global_step_<N>/                  # checkpoints
├── <N>/<video_id>.json               # per-step, per-video reward log
└── history/<video_id>.json           # accumulated memory used as preceding-context
                                      # for subsequent curriculum steps
```

W&B is enabled by default (`trainer.logger=["console","wandb"]`,
`trainer.project_name=m3_qwen3_vl`). Override `trainer.logger=["console"]`
to disable.
