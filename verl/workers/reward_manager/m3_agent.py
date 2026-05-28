# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import base64
import json
import os
import random
import re
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
from time import sleep
from typing import Any

import numpy as np
import openai
import torch
from json_repair import loads, repair_json
from torch.nn.utils.rnn import pad_sequence

from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

MAX_WORKERS = 4
LLM_TEMPERATURE = 1e-6
MAX_RETRIES = 5
FACE_PATTERN = re.compile(r"\bface[ _]\d+\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Azure/OpenAI client loader
# ---------------------------------------------------------------------------
#
# Path to the Azure OpenAI endpoint/key config. The JSON has the schema:
# {
#   "gpt-4o": {"azure_endpoint": "...", "api_version": "...", "api_key": "..."},
#   "gpt-4o-mini": [{"azure_endpoint": "...", "api_version": "...", "api_key": "..."}, ...]
# }
# See ``configs/api_config.example.json`` for a template.

_API_CLIENT_KEYS = ("azure_endpoint", "api_version", "api_key")


def _load_api_clients(config_path: str):
    with open(config_path) as f:
        config = json.load(f)
    clients = {}
    for model_name, conf in config.items():
        if isinstance(conf, list):
            clients[model_name] = [
                openai.AzureOpenAI(**{k: c[k] for k in _API_CLIENT_KEYS}) for c in conf
            ]
        else:
            clients[model_name] = openai.AzureOpenAI(
                **{k: conf[k] for k in _API_CLIENT_KEYS}
            )
    return clients, config


_API_CONFIG_PATH = os.environ.get("M3_API_CONFIG", "configs/api_config.json")
client, config = _load_api_clients(_API_CONFIG_PATH)


def timeout(seconds=300, default=None, raise_err=True):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = [TimeoutError(f"Function '{func.__name__}' timed out after {seconds}s")]
            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    result[0] = e
            thread = threading.Thread(target=target)
            thread.start()
            thread.join(seconds)
            if thread.is_alive():
                if raise_err:
                    raise TimeoutError(f"Function '{func.__name__}' timed out after {seconds}s")
                return default
            if isinstance(result[0], Exception):
                raise result[0]
            return result[0]
        return wrapper
    return decorator

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_PROMPT_EPISODIC_CORRECTNESS = """You are provided with a video, a description of its preceding segment, and a generated candidate [Description] for the remaining portion.
Your task is to evaluate:
1. Whether the candidate description is factually accurate based only on visual content and subtitles (ignore audio).
2. Whether it connects coherently and naturally with the preceding description, without using transition words such as "continue".
For any spoken content, verify it solely against the displayed subtitles and disregard audio information.
Assign exactly one label:
1: Correct — The description that meets all of the above criteria.
0: Incorrect — Any description that fails to meet the above criteria.

Output Requirements: Return the result in the following valid JSON format only. Do not generate anything else.
{{
    "correctness_rationale": "Short explanation for marking this description as 1 or 0",
    "correctness": 1 or 0 
}}

The description of the preceding segment:
{preceding_json}

The [Description] to verify:
{blocks_text}
""".strip()

_PROMPT_EPISODIC_LABEL = """
You are given the [Context] and a candidate description that are describing new events.

Your task is to evaluate whether the candidate description satisfies the following conditions.

Return label=0 if any condition is satisfied, else 1:
(1) The description repeats any atomic fact already present in the [Context].
(2) It includes any mention of bounding boxes, coordinates, or detection boxes (e.g., "bounding box", "bbox", "x1,y1,x2,y2", "rectangle box around").
(3) It contains meta phrases like: "subtitles said", "the subtitles say", "subtitle reads", "subtitle says", or "according to the subtitles".
(4) The quoted speech contains transcript-style speaker labels like "<face_id> says "<face_id>: Good"" inside quoted dialogue.
(5) It includes conclusion-based or context-setting statements such as "this video ends with..." or "based on previous videos".

Output Requirements: Return the result in the following valid JSON format only. Do not generate anything else.

{{
    "label_rationale": "Short explanation for marking this description as 1 or 0",
    "label": 1 or 0
}}

[Context]:
{preceding_json}

candidate description to verify:
{blocks_text}

""".strip()

_PROMPT_EPISODIC_USEFULNESS = """You are given a list of descriptions summaried from a video, each associated with a unique ID. Please rank these descriptions based on their usefulness, output their ID from high to low. Usefulness should be determined by the amount of non-redundant, unique information contained in each item; items with more unique and less overlapping information should be ranked higher. Besides, descriptions that include dialogue directly in the narrative (e.g., <face_id> said, "xxx") should be ranked higher than descriptions that reference dialogue by referencing subtitles, captions, or other UI elements. The length of the output list must match the input list exactly.

Output format:
[RANK START]
[2, 1, 3, 6, 4, 5]
[RANK END]

Input Knowledge:
{blocks_text}

Output the list of ID:"""

PROMPTS = {
    "episodic_correctness": _PROMPT_EPISODIC_CORRECTNESS,
    "episodic_label": _PROMPT_EPISODIC_LABEL,
    "episodic_usefulness": _PROMPT_EPISODIC_USEFULNESS,
}


def get_response_with_retry(model, messages, timeout=30):
    for _ in range(MAX_RETRIES):
        try:
            if isinstance(client[model], list):
                selected_model = random.choice(client[model])
            else:
                selected_model = client[model]
            if model in ["gemini-2.5-flash", "gemini-2.5-pro"]:
                extra_body={
                    "thinking": {
                        "include_thoughts": True,
                        "budget_tokens": 128
                    }
                }
                response = selected_model.chat.completions.create(model=model, messages=messages, temperature=LLM_TEMPERATURE, timeout=timeout, extra_body=extra_body, max_tokens=8192)
            else:
                response = selected_model.chat.completions.create(model=model, messages=messages, temperature=LLM_TEMPERATURE, timeout=timeout, max_tokens=8192)
            return response.choices[0].message.content
        except Exception as e:
            print("Failed to get response:", e)
            sleep(5)
            continue
    raise Exception(f"Failed to get response after {MAX_RETRIES} retries")

def generate_messages(inputs):
    messages = []
    messages.append(
        {"role": "system", "content": "You are an expert in video understanding."}
        )
    content = []
    for input in inputs:
        if input["type"] == "text":
            content.append(input)
        elif input["type"] == "video":
            base64_video = base64.b64encode(open(input["video"], "rb").read()).decode("utf-8")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:video/mp4;base64,{base64_video}"},
                }
                )
        else:
            raise ValueError(f"Invalid input type: {input['type']}")
    messages.append({"role": "user", "content": content})
    return messages

def _to_jsonable(x):
    if isinstance(x, np.ndarray):
        return [_to_jsonable(v) for v in x.tolist()]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (list, tuple, set)):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    # Optional: make bytes printable
    if isinstance(x, (bytes, bytearray)):
        return x.decode("utf-8", errors="replace")
    return x  # already JSON-friendly

def _atomic_json_dump(payload, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(payload), f, ensure_ascii=False, indent=2)


class MultiRewardEvaluator:
    def __init__(
        self,
        model_name_text="gpt-4o-2024-11-20",
        model_name_video="gemini-2.5-flash",
        tokenizer=None,
        max_response_length=8196,
        think_length_threshold=1600,
        think_penalty=-1.0,
        memory_length_threshold=5,
        usefulness_scale=1.0,
        memory_token_threshold=25,
        format_penalty=-3.0,
        correct_reward=0.5,
        wrong_reward=-0.2,
        failure_penalty=0.0,
    ):
        self.model_name_text = model_name_text
        self.model_name_video = model_name_video
        self.think_penalty = think_penalty
        self.think_start_ids = [151667,]
        self.think_end_ids = [151668,]
        self.memory_length_threshold = memory_length_threshold
        self.think_length_threshold = think_length_threshold
        self.max_response_length = max_response_length
        self.memory_token_threshold = memory_token_threshold
        self.correct_reward = correct_reward
        self.wrong_reward = wrong_reward
        self.format_penalty = format_penalty
        self.tokenizer = tokenizer
        self.usefulness_scale = usefulness_scale
        self.failure_penalty = failure_penalty

        self.prompts = {
            ("episodic", "correctness"): PROMPTS["episodic_correctness"],
            ("episodic", "usefulness"): PROMPTS["episodic_usefulness"],
            ("episodic", "label"): PROMPTS["episodic_label"]
        }

        self.pad_id = self.tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = self.tokenizer.eos_token_id

    def find_pattern(self, seq, pat):
        if isinstance(pat, int) or (isinstance(pat, list) and len(pat) == 1):
            target = pat if isinstance(pat, int) else pat[0]
            seq_tensor = torch.tensor(seq)
            return (seq_tensor == target).nonzero(as_tuple=True)[0].tolist()

        # fallback to slow version for multi-token patterns
        n, m = len(seq), len(pat)
        return [i for i in range(n - m + 1) if seq[i:i + m] == list(pat)]
    def get_safe_values(self, source_dict, key, default_val=None):
        val = source_dict.get(key, {})
        if isinstance(val, dict):
            return list(val.values())
        print(f"[Error] Key '{key}' expected dict but got {type(val)}. Content: {str(val)[:100]}")
        return default_val if default_val is not None else []
    def right_trim(self, ids, pad_token_id):
        i = len(ids)
        while i > 0 and ids[i - 1] == pad_token_id:
            i -= 1
        return ids[:i]
    def _invalid_face_tag(self, desc: str) -> bool:
        if re.search(r'<face>', desc, re.IGNORECASE):
            return True

        suspicious_pattern = re.compile(r'face[\s_]*\d+', re.IGNORECASE)

        for m in suspicious_pattern.finditer(desc):
            s, e = m.span()
            token = m.group()

            if token != f"face_{token.split('_')[-1]}" or not re.fullmatch(r'face_\d+', token):
                if not re.fullmatch(r'face_\d+', token):
                    return True

            has_brackets = (s > 0 and desc[s-1] == '<' and e < len(desc) and desc[e] == '>')
            if not has_brackets:
                return True

        return False

    def compute_think_reward(self, think_len: int) -> float:
        L = self.think_length_threshold
        if 0 <= think_len <= L:
            return 0.0
        if think_len >= 2*L:
            return self.think_penalty
        return self.think_penalty * (think_len - L) / L

    @staticmethod
    def _make_default_info(
        i: int,
        mode: str,
        ids,
        think_len: int,
        descs,
        spans,
        txt,
        preceding_description,
        memory_length_threshold: int = None,
    ) -> dict:
        if descs is None:
            eval_descs = []
        elif memory_length_threshold is not None:
            eval_descs = descs[:memory_length_threshold]
        else:
            eval_descs = descs
        return {
            "group": i,
            "mode": mode,
            "ids": ids,
            "think_len": think_len,
            "memory_token_length": max(0, len(ids) - think_len),
            "group_type": None,
            "num_correct": 0,
            "num_wrong": 0,
            "fail_parsing": 0 if descs is not None else 1,
            "invalid_face_list": [],
            "correctness_list": [],
            "accuracy_list": [],
            "ori_correctness_list": [],
            "redundancy_list": [],
            "usefulness_list": [],
            "valid_list": [],
            "memory_token_list": [],
            "has_face_list": [],
            "cot_correctness": [],
            "cot_redundancy": [],
            "is_format_error": False,
            "has_invalid_face": False,
            "r_think": 0,
            "output_memory": descs if descs is not None else [],
            "descs": descs,
            "eval_descs": eval_descs,
            "spans": spans,
            "txt": txt,
            "preceding_description": preceding_description if preceding_description is not None else "",
        }

    def compute_token_level_rewards(
        self,
        token_id_matrix,
        mode="episodic",
        extra_info=None,
        timeout=480,
    ):
        extra_info = extra_info or {}
        video_path = extra_info.get("video_path")
        preceding_description = extra_info.get("preceding_description", "")

        if hasattr(token_id_matrix, "detach"):
            token_id_matrix = token_id_matrix.detach().cpu().tolist()

        groups, decoded_info = self._decode_and_parse(
            token_id_matrix=token_id_matrix,
            mode=mode,
            video_path=video_path,
            preceding_description=preceding_description,
        )

        if groups:
            results = self._evaluate_all_tasks(groups, mode, timeout)
            result_map = self._flatten_results(results)
        else:
            result_map = {}

        invalid_face_rids = self._collect_invalid_face_rids(decoded_info)

        sample_info_list = self._build_sample_infos(
            decoded_info=decoded_info,
            mode=mode,
            result_map=result_map,
            invalid_face_rids=invalid_face_rids,
            preceding_description=preceding_description,
        )

        sample_info_list = self._stage1_score(sample_info_list, mode, result_map)
        return self._pack_rewards(token_id_matrix, sample_info_list)

    def _decode_and_parse(
        self,
        token_id_matrix,
        mode,
        video_path,
        preceding_description,
    ):
        groups, decoded_info = [], []
        for i, original_ids in enumerate(token_id_matrix):
            valid_ids = self.right_trim(list(original_ids), self.pad_id)
            txt = self.tokenizer.decode(original_ids, skip_special_tokens=True)

            start_think_i = self.find_pattern(valid_ids, self.think_start_ids)
            end_think_i = self.find_pattern(valid_ids, self.think_end_ids)
            if not end_think_i or len(end_think_i) > 1:
                decoded_info.append((valid_ids, self.max_response_length, None, (0, len(valid_ids)), txt, []))
                print(f"[Warning] No thinking parsing found in group {i}")
                continue

            if start_think_i and end_think_i:
                think_len = max(0, end_think_i[0] - (start_think_i[0] + len(self.think_start_ids)))
            else:
                think_len = max(0, end_think_i[0])
            think_idx = (start_think_i[0] if start_think_i else 0, end_think_i[0])

            try:
                memory_key = "description"
                memory_desc = json.loads(txt.split("</think>")[-1].strip().strip("```json").strip())
                if (
                    not isinstance(memory_desc, dict)
                    or memory_key not in memory_desc
                    or len(memory_desc) > 1
                    or len(memory_desc[memory_key]) == 0
                ):
                    decoded_info.append((valid_ids, self.max_response_length, None, think_idx, txt, []))
                    continue
                val = memory_desc[memory_key]
                descs = [val if isinstance(val, str) else str(val)]
                spans = [(think_idx[1], len(valid_ids))]
            except Exception:
                print(f"Error parsing JSON for group {i}")
                decoded_info.append((valid_ids, self.max_response_length, None, think_idx, txt, []))
                continue

            if len(spans) != len(descs) or len(descs) == 0 or descs[0] == "":
                decoded_info.append((valid_ids, think_len, None, think_idx, txt, spans))
                continue

            groups.append({
                "group_id": f"{i}",
                mode: descs[:self.memory_length_threshold],
                "video_path": video_path,
                "preceding_description": preceding_description,
            })
            decoded_info.append((valid_ids, think_len, descs, think_idx, txt, spans))

        return groups, decoded_info

    def _collect_invalid_face_rids(self, decoded_info) -> set:
        invalid_face_rids = set()
        for i, (_, _, descs, _, _, _) in enumerate(decoded_info):
            if descs is None:
                continue
            gid = f"{i}"
            for j, d in enumerate(descs):
                if self._invalid_face_tag(d):
                    invalid_face_rids.add(f"{gid}_{j+1}")
        return invalid_face_rids

    def _build_sample_infos(
        self,
        decoded_info,
        mode,
        result_map,
        invalid_face_rids,
        preceding_description,
    ):
        sample_info_list = []
        for i, (ids, think_len, descs, _, txt, spans) in enumerate(decoded_info):
            gid = f"{i}"
            info = self._make_default_info(
                i=i,
                mode=mode,
                ids=ids,
                think_len=think_len,
                descs=descs,
                spans=spans,
                txt=txt,
                preceding_description=preceding_description,
                memory_length_threshold=self.memory_length_threshold,
            )
            eval_descs = info["eval_descs"]
            if descs is None:
                info["group_type"] = "parse_error"
                info["is_format_error"] = True
                info["fail_parsing"] = 1
                sample_info_list.append(info)
                continue


            for d in descs:
                info["has_face_list"].append(1 if FACE_PATTERN.search(d) else 0)

            for j in range(len(descs)):
                if f"{gid}_{j+1}" in invalid_face_rids:
                    info["has_invalid_face"] = True
                    break

            task_results = result_map.get(gid, {})
            correctness_key = "correctness"
            coh_key = "label"
            if correctness_key not in task_results or coh_key not in task_results:
                info["group_type"] = "reward_error"
                info["fail_parsing"] = 2
                sample_info_list.append(info)
                continue

            corr_vals = self.get_safe_values(task_results, correctness_key, default_val=[0]*len(eval_descs))
            coh_vals = self.get_safe_values(task_results, coh_key, default_val=[0]*len(eval_descs))
            if len(corr_vals) != len(eval_descs) or len(coh_vals) != len(eval_descs):
                info["group_type"] = "reward_error"
                info["fail_parsing"] = 2
                sample_info_list.append(info)
                continue

            corr_cot = self.get_safe_values(task_results, f"{correctness_key}_rationale", default_val=[""]*len(eval_descs))
            corr_cot += [""] * (len(descs) - len(corr_cot))
            cot_red = self.get_safe_values(task_results, f"{coh_key}_rationale", default_val=[""]*len(eval_descs))
            cot_red += [""] * (len(descs) - len(cot_red))

            assert len(spans) == len(descs), f"spans: {spans}, descs: {descs}"
            if len(eval_descs) == 0:
                info["redundancy_list"].append(0)
                info["correctness_list"].append(0)
                info["num_correct"] = 0

            for j, _ in enumerate(descs):
                rid = f"{gid}_{j+1}"
                token_len = spans[j][1] - spans[j][0] if j < len(spans) else 0
                info["memory_token_list"].append(token_len)
                invalid_face = rid in invalid_face_rids
                info["invalid_face_list"].append(1 if invalid_face else 0)

                if j < len(eval_descs):
                    acc_valid = not (invalid_face or token_len > self.memory_token_threshold)
                    is_valid = acc_valid and coh_vals[j] != 0

                    is_correct = (float(corr_vals[j]) > 0.5 and is_valid)
                    acc_val = acc_valid and float(corr_vals[j]) > 0

                    info["cot_correctness"].append(corr_cot[j])
                    info["correctness_list"].append(float(corr_vals[j]) if is_correct else 0)
                    info["accuracy_list"].append(1 if acc_val else 0)
                    info["ori_correctness_list"].append(corr_vals[j])

                    info["cot_redundancy"].append(cot_red[j])
                    info["redundancy_list"].append(coh_vals[j])
                    info["valid_list"].append(float(is_correct and coh_vals[j]))
                else:
                    info["correctness_list"].append(0)
                    info["redundancy_list"].append(0)
                    info["accuracy_list"].append(0)
                    info["valid_list"].append(0)

            info["num_correct"] = sum(info["correctness_list"])
            info["num_wrong"] = max(0, len(descs) - info["num_correct"])
            info["group_type"] = "normal"

            sample_info_list.append(info)
        return sample_info_list

    def _pack_rewards(self, token_id_matrix, sample_info_list):
        all_rewards, all_masks, reward_logs = [], [], []
        for i, info in enumerate(sample_info_list):
            original_len = len(token_id_matrix[i])
            valid_len = len(info["ids"])

            reward = torch.zeros(original_len, dtype=torch.float32)
            mask = torch.zeros(original_len, dtype=torch.int)
            last_valid_pos = valid_len - 1 if valid_len > 0 else 0

            ids_row = torch.tensor(token_id_matrix[i])
            assert ids_row[last_valid_pos].item() != self.pad_id, "reward on pad token"
            nonpad_len = int((ids_row != self.pad_id).sum().item())
            assert valid_len <= nonpad_len + 1, f"valid_len too large: {valid_len} vs {nonpad_len}"

            if original_len > 0 and last_valid_pos >= 0:
                reward[last_valid_pos] = info["final_score"]
                mask[last_valid_pos] = 1

            all_rewards.append(reward)
            all_masks.append(mask)
            del info["descs"], info["spans"], info["ids"]
            reward_logs.append(info)

        padded_rewards = pad_sequence(all_rewards, batch_first=True, padding_value=0.0)
        all_masks_tensor = pad_sequence(all_masks, batch_first=True, padding_value=0)
        return padded_rewards, reward_logs, all_masks_tensor

    def _eval_one_group(self, group, mode, task_type, timeout=480):
        tmpl = self.prompts.get((mode, task_type))
        if not tmpl:
            return {task_type: {}, f"{task_type}_responses": ""}

        gid = group["group_id"]
        descs = group.get(mode, [])
        if not descs:
            return {
                task_type: {
                    gid: {f"{task_type}_rationale": {}, task_type: {}}
                },
                f"{task_type}_responses": ""
            }

        blocks_text = descs[0]
        if not blocks_text:
            return {task_type: {}, f"{task_type}_responses": ""}

        prompt = tmpl.format(
            preceding_json=group.get("preceding_description", ""),
            blocks_text=blocks_text,
        )

        if task_type == "correctness":
            inputs = [
                {"type": "video", "video": group["video_path"]},
                {"type": "text", "text": prompt},
            ]
            model_name = self.model_name_video
        else:
            inputs = [{"type": "text", "text": prompt}]
            model_name = self.model_name_text

        result = self._process_task(inputs, task_type, model_name, timeout)

        parsed = result.get(task_type, {})
        return {
            task_type: {gid: parsed},
            f"{task_type}_responses": result.get(f"{task_type}_responses", "")
        }

    def _rank_descs_by_usefulness(self, descs, mode, write_key="usefulness", timeout=480):
        if not descs:
            return list(range(len(descs))), {f"{write_key}_responses": "", f"{write_key}_position": [], f"{write_key}_memory": ""}

        n = len(descs)
        perm = list(range(n))
        random.shuffle(perm)

        input_text = "\n".join(f"{i+1}. {descs[idx]}" for i, idx in enumerate(perm))
        tmpl = self.prompts.get((mode, "usefulness"))
        prompt = tmpl.format(blocks_text=input_text)
        inputs = [{"type": "text", "text": prompt}]
        messages = generate_messages(inputs)

        last_error = None
        last_responses = {"usefulness_responses": ""}
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                responses = get_response_with_retry(self.model_name_text, messages, timeout)
                last_responses["usefulness_responses"] = responses
                match = re.search(r"\[RANK START\](.*?)\[RANK END\]", responses, re.S)
                if not match:
                    raise ValueError("No [RANK START]/[RANK END] block found in response.")

                parsed_rank = json.loads(match.group(1).strip())
                if not isinstance(parsed_rank, list):
                    raise ValueError("Parsed rank is not a list.")

                order = []
                for display_id in parsed_rank:
                    if isinstance(display_id, int) and 1 <= display_id <= n:
                        original_idx = perm[display_id - 1]
                        if original_idx not in order:
                            order.append(original_idx)

                if len(order) != len(descs) and attempt < MAX_RETRIES:
                    raise ValueError("Order length mismatch.")

                return order, {f"{write_key}_responses": responses, f"{write_key}_position": perm, f"{write_key}_memory": input_text}
            except Exception as e:
                last_error = e
        print(f"[EvalLLM] usefulness ranking failed: {last_error}")
        return list(range(len(descs))), last_responses

    _TASK_DISPATCH = {
        "correctness": "_eval_llm",
        "label": "_eval_label_episodic",
    }

    def _evaluate_all_tasks(self, groups, mode, timeout=480):
        tasks = list(self._TASK_DISPATCH.keys())
        task_results = {}

        def run_task(task_type):
            handler = getattr(self, self._TASK_DISPATCH[task_type])
            return handler(groups, mode, task_type, timeout)

        with ThreadPoolExecutor(max_workers=len(tasks)) as ex:
            future_to_type = {ex.submit(run_task, t): t for t in tasks}
            for fut in as_completed(future_to_type):
                task_type = future_to_type[fut]
                try:
                    task_results[task_type] = fut.result()
                except Exception as e:
                    print(f"[EvalTask] {task_type} failed: {e}")
                    task_results[task_type] = None

        return [task_results[t] or {} for t in tasks]

    def _eval_llm(self, groups, mode, task_type, timeout=480):
        tmpl = self.prompts.get((mode, task_type))
        if not tmpl:
            return {task_type: {}}

        merged_results = {}
        merged_responses = {}

        worker_num = min(len(groups), 4)

        def run_one_group(g):
            gid = g["group_id"]
            try:
                result = self._eval_one_group(
                    g, mode, task_type, timeout=timeout
                )
                return gid, result.get(task_type, {}).get(gid, {}), result.get(f"{task_type}_responses", "")
            except Exception as e:
                print(f"[EvalLLM] {task_type} for group {gid} failed: {e}")
                return gid, {}, ""

        with ThreadPoolExecutor(max_workers=worker_num) as ex:
            futures = [ex.submit(run_one_group, g) for g in groups]
            for fut in as_completed(futures):
                gid, one_result, one_response = fut.result()
                merged_results[gid] = one_result
                merged_responses[gid] = one_response

        return {
            task_type: merged_results,
            f"{task_type}_responses": json.dumps(merged_responses, ensure_ascii=False, indent=2),
        }

    @timeout(600, default={}, raise_err=False)
    def _eval_label_episodic(self, groups, mode, task_type, timeout=480):
        tmpl = self.prompts.get((mode, task_type))
        if not tmpl:
            return {task_type: {}}
        results = {}
        responses = {}
        for g in groups:
            gid = g["group_id"]
            descs = g[mode]
            if descs and descs[0]:
                prompt = tmpl.format(
                    preceding_json=groups[0].get("preceding_description", ""),
                    blocks_text=descs[0],
                )
                inputs = [{"type": "text", "text": prompt}]
                try:
                    results_g = self._process_task(inputs, task_type, self.model_name_text, timeout)
                    results[gid] = results_g.get(task_type, {})
                    responses[gid] = results_g.get(f"{task_type}_responses", "")
                except Exception as e:
                    print(f"[EvalLLM] {task_type} for group {gid} failed: {e}")
                    results[gid] = {f"{task_type}_rationale": {}, f"{task_type}": {}}
            else:
                results[gid] = {f"{task_type}_rationale": {}, f"{task_type}": {}}
                continue
        return {task_type: results, f"{task_type}_responses": json.dumps(responses, indent=2)}

    def _process_task(self, task_inputs, task_type, model_name, timeout):
        last_responses = ""
        for attempt in range(1, 20):
            try:
                messages = generate_messages(task_inputs)
                responses = get_response_with_retry(model_name, messages, timeout)
                last_responses = responses
                llm_responses = self._parse_llm_response(responses, task_type)
                return {task_type: llm_responses, f"{task_type}_responses": responses}
            except Exception as e:
                print(f"[EvalLLM] {task_type} failed: {e} (attempt {attempt})")
                last_error = e
        print(f"[EvalLLM] {task_type} failed: {last_error}")
        return {task_type: {}, f"{task_type}_responses": last_responses}

    def _validate_score_value(self, v):
        if isinstance(v, bool):
            return True
        if isinstance(v, (int, float)):
            return True
        if isinstance(v, str):
            try:
                float(v.strip())
                return True
            except Exception:
                return False
        return False

    def _parse_llm_response(self, responses, task_type):
        parsed = loads(repair_json(responses.strip().strip("`json").strip()))
        score_key = task_type

        if not isinstance(parsed, dict):
            raise ValueError("Missing or invalid JSON object.")
        if score_key not in parsed:
            raise ValueError(f"missing '{score_key}' in response")
        if not self._validate_score_value(parsed[score_key]):
            raise ValueError(f"not int value: {parsed}")
        return {
            score_key: {"1": parsed[score_key]},
            f"{score_key}_rationale": {"1": parsed.get(f"{score_key}_rationale", "")},
        }

    def _flatten_results(self, results):
        merged = {}
        for r in results:
            for k, v in r.items():
                if isinstance(v, dict):
                    for gid, data in v.items():
                        merged.setdefault(gid, {}).update(data)
                else:
                    merged.setdefault("global", {})[k] = v
        return merged

    def _stage1_score(
        self,
        sample_info_list: list,
        mode: str,
        result_map: dict,
    ):
        for info in sample_info_list:
            descs = info.get("descs") or []
            info["usefulness_list"] = [0.0] * len(descs)
            info["traj_usefulness_raw"] = 0.0
            info["traj_usefulness_bonus"] = 0.0

        global_descs = []
        global_owner = []
        for s_idx, info in enumerate(sample_info_list):
            if info.get("fail_parsing", 0):
                continue
            descs = info.get("descs") or []
            corr  = info.get("correctness_list") or []
            for j, (d, c) in enumerate(zip(descs, corr, strict=False)):
                if c > 0:
                    global_owner.append((s_idx, j))
                    global_descs.append(d)

        if global_descs:
            order, usefulness_responses = self._rank_descs_by_usefulness(global_descs, mode)
            result_map.setdefault("global", {}).update(usefulness_responses)

            M = len(global_descs)
            if order:
                denom = (len(order) - 1) if len(order) > 1 else 1
                assigned = {gidx: (1.0 - pos / denom) for pos, gidx in enumerate(order)}
                mean_score = sum(assigned.values()) / len(order)
                global_scores = [assigned.get(i, mean_score) for i in range(M)]
            else:
                global_scores = [0.0] * M

            for gidx, (s_idx, j) in enumerate(global_owner):
                sample_info_list[s_idx]["usefulness_list"][j] = float(global_scores[gidx])

        x_vals = []
        for info in sample_info_list:
            if info["fail_parsing"]:
                continue
            corr = info.get("correctness_list") or []
            correct_js = [j for j, c in enumerate(corr) if c > 0]
            if not correct_js:
                info["traj_usefulness_raw"] = 0.0
                continue
            vals = [info["usefulness_list"][j] for j in correct_js]
            info["traj_usefulness_raw"] = float(sum(vals) / len(vals))
            x_vals.append(info["traj_usefulness_raw"])

        x_min = min(x_vals) if x_vals else 0.0
        x_max = max(x_vals) if x_vals else 0.0
        if (x_max - x_min) > 1e-8:
            for info in sample_info_list:
                if info["fail_parsing"]:
                    continue
                corr = info.get("correctness_list") or []
                if any(c > 0 for c in corr):
                    x = info["traj_usefulness_raw"]
                    info["traj_usefulness_bonus"] = float(
                        self.correct_reward * (x - x_min) / (x_max - x_min)
                    )
        valid_item_scores = []
        for info in sample_info_list:
            if info["fail_parsing"] == 1:
                s1 = float(self.format_penalty)
                info["r_task"] = s1
                info["r_think"] = 0.0
            elif info["fail_parsing"] == 2:
                s1 = 0
            else:
                descs = info.get("descs") or []
                corr = info.get("correctness_list") or []
                s1 = 0.0
                for j in range(len(descs)):
                    c = corr[j]
                    s1 += c * self.correct_reward + (c == 0) * self.wrong_reward

                s1 += float(self.usefulness_scale * info["traj_usefulness_bonus"])
                info["r_task"] = s1
                info["r_think"] = float(self.compute_think_reward(info["think_len"]))
                s1 += info["r_think"]

            info["final_score"] = float(s1)
            if info["fail_parsing"] == 0:
                valid_item_scores.append(s1)
            info["global_results"] = result_map.get("global", {})

        min_valid_score = np.min(valid_item_scores) if len(valid_item_scores) else 0.0
        for info in sample_info_list:
            if info["fail_parsing"] == 1 and self.failure_penalty != 0.0:
                info["final_score"] = float(min_valid_score) + self.failure_penalty

        return sample_info_list

@register("m3_agent")
class M3AgentRewardManager(AbstractRewardManager):
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", **reward_kwargs) -> None:
        """
        Initialize the NaiveRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source
        self.reward_evaluator = MultiRewardEvaluator(tokenizer=self.tokenizer, **reward_kwargs)
        self.max_response_length = reward_kwargs.get("max_response_length")

    def _split_history_id(self, raw_id: Any):
        s = str(raw_id).strip()
        if s.endswith(".json"):
            s = s[:-5]

        if "*" in s:
            file_stem, key = s.split("*", 1)
        else:
            file_stem, key = s, "0"

        return file_stem, key

    def _pick_one_memory(self, reward_logs_for_group: list):
        default_mem = ""
        for rl in reward_logs_for_group:
            if not isinstance(rl, dict):
                continue
            if rl.get("fail_parsing", 1) in [0, 2]:
                out = rl.get("output_memory", None)
                if isinstance(out, list) and len(out) > 0:
                    return out[0]
                return default_mem
        return default_mem

    def __call__(self, data, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        batch = data.batch
        non_tensor = data.non_tensor_batch
        responses = batch["responses"]
        meta_info = getattr(data, "meta_info", {}) or {}
        root_path = meta_info.get("root_path", os.environ.get("M3_CKPT_ROOT", "./ckpts/m3"))
        global_step = str(meta_info.get("global_steps", 0))
        file_dir = os.path.join(root_path, global_step)
        os.makedirs(file_dir, exist_ok=True)

        types = non_tensor["type"]
        inputs = non_tensor["input"]
        uids = non_tensor["uid"]
        video_ids = non_tensor["id"].tolist()

        rank = torch.distributed.get_rank() if (torch.distributed.is_available() and torch.distributed.is_initialized()) else 0
        B = responses.size(0)
        batch["trajectory_uid"] = rank * 1_000_000 + torch.arange(B, device=responses.device, dtype=torch.long)

        uid2id = {}
        uid_ids = []
        for u in uids:
            key = u if isinstance(u, (str, int)) else str(u)
            if key not in uid2id:
                uid2id[key] = len(uid2id)
            uid_ids.append(uid2id[key])

        batch["uid"] = torch.tensor(uid_ids, device=responses.device, dtype=torch.long)

        # Step 1: group by (id, video_id, type)
        group_map = defaultdict(list)
        for i, (uid, video_id, typ) in enumerate(zip(uids, video_ids, types, strict=False)):
            group_map[(uid, video_id, typ)].append(i)

        final_rewards = torch.zeros_like(responses, dtype=torch.float32)
        final_masks = torch.zeros_like(responses, dtype=torch.int)
        reward_log_list = [
            {}
            for i in range(len(responses))
        ]
        def compute_for_group(typ, indices):
            token_id_matrix = responses[indices]
            input_blocks = inputs[indices[0]]

            extra_info = self._extract_extra_info(input_blocks)
            return (
                indices,
                self.reward_evaluator.compute_token_level_rewards(
                    token_id_matrix,
                    mode=typ,
                    extra_info=extra_info,
                )
            )

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(compute_for_group, typ, indices): (vid, video_id, typ)
                for (vid, video_id, typ), indices in group_map.items()
            }

            for f in as_completed(futures):
                vid, video_id, typ = futures[f]
                safe = str(video_id)
                file_path = os.path.join(file_dir, f"{safe}.json")
                existing = {}
                if os.path.exists(file_path):
                    try:
                        with open(file_path, "r", encoding="utf-8") as fr:
                            existing = json.load(fr)
                    except Exception:
                        existing = {}
                        print("existing file is broken")

                indices, (reward_tensor, reward_logs, mask_tensor) = f.result()
                assert reward_tensor.shape[0] == mask_tensor.shape[0], \
                    f"Mask/reward batch mismatch: {reward_tensor.shape[0]} vs {mask_tensor.shape[0]}"
                assert reward_tensor.shape[0] == len(reward_logs), \
                    f"Reward logs length mismatch: {reward_tensor.shape[0]} vs {len(reward_logs)}"

                reward_tensor = reward_tensor.to(final_rewards.device, non_blocking=True)
                mask_tensor = mask_tensor.to(final_masks.device, non_blocking=True)
                seq_len = min(reward_tensor.size(1), responses.size(1))

                group_items = []
                for j, idx in enumerate(indices):
                    final_rewards[idx, :seq_len] = reward_tensor[j, :seq_len]
                    final_masks[idx, :seq_len] = mask_tensor[j, :seq_len]
                    reward_log_list[idx] = reward_logs[j]

                    reward_logs[j]["trajectory_uid"] = batch["trajectory_uid"][idx].item()
                    group_items.append({
                        "input": inputs[idx],
                        "reward": reward_logs[j] if reward_logs is not None else {},
                    })

                group_obj = {
                    "id": vid,
                    "type": typ,
                    "global_step": global_step,
                    "num_items": len(group_items),
                    "items": group_items,
                }
                self._persist_group_payload(group_obj, existing, file_path)
                self._persist_group_history(reward_logs, video_id, root_path)


        if return_dict:
            return {
                "reward_tensor": final_rewards,
                "mask_tensor": final_masks,
                "reward_extra_info": {
                    "reward_log": reward_log_list,
                },
            }
        return final_rewards, final_masks

    def _persist_group_payload(self, group_obj: dict, existing: dict, file_path: str) -> None:
        if isinstance(existing, dict) and "items" in existing:
            merged = dict(existing)
            merged.setdefault("id", group_obj["id"])
            merged.setdefault("type", group_obj["type"])
            merged.setdefault("global_step", group_obj["global_step"])
            merged_items = (existing.get("items") or []) + group_obj["items"]
            merged["items"] = merged_items
            merged["num_items"] = len(merged_items)
            payload = merged
        else:
            payload = group_obj
        _atomic_json_dump(payload, file_path)

    def _persist_group_history(self, reward_logs, video_id, root_path: str) -> None:
        history_dir = os.path.join(root_path, "history")
        os.makedirs(history_dir, exist_ok=True)

        file_stem, hist_key = self._split_history_id(video_id)
        history_path = os.path.join(history_dir, f"{file_stem}.json")
        one_memory = self._pick_one_memory(reward_logs)

        existing_hist = {}
        if os.path.exists(history_path):
            try:
                with open(history_path, "r", encoding="utf-8") as fr:
                    existing_hist = json.load(fr)
                    if not isinstance(existing_hist, dict):
                        existing_hist = {}
            except Exception:
                existing_hist = {}

        existing_hist[str(hist_key)] = one_memory
        _atomic_json_dump(existing_hist, history_path)

    def _extract_extra_info(self, input_list):
        video_path, preceding = None, []
        text_items = [it["text"] for it in input_list if it.get("type") == "text" and "text" in it]
        last_text = text_items[-1] if text_items else ""
        preceding = last_text.split("[Description of the preceding part]:")[1].split("\n\n- Generate subsequent descriptions not covered in [Description of the preceding part]")[0].strip()
        preceding = preceding if isinstance(preceding, str) else ""
        for item in input_list:
            if item["type"] == "video":
                video_path = item["video"]
                break
        return {
            "video_path": video_path,
            "preceding_description": preceding,
        }
