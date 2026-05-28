import logging
import json
import copy
import re
import os
import numpy as np
from typing import Optional
from qwen_vl_utils import process_vision_info
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import ProcessorMixin
from verl.models.transformers.qwen3_vl import get_rope_index
import verl.utils.torch_functional as verl_F
from typing import List, Dict, Any
logger = logging.getLogger(__name__)

def unify_face_id(text):
    pattern = r'([<\(]?)(face)([\s_]*)(\d+)([>\)]?)'
    replaced_text = re.sub(
        pattern,
        lambda match: f"<{match.group(2).lower()}_{match.group(4)}>",  # group(1) 转小写
        text,
        flags=re.IGNORECASE  # 忽略大小写匹配
    )
    return replaced_text

class VLDataset(Dataset):
    """
    Load and preprocess RLHF data from Parquet files.

    - Caches files locally.
    - Reads into a HuggingFace Dataset and tokenizes prompts.
    - Optionally handles images/videos via a ProcessorMixin.
    - Filters prompts over a max length.
    - Supports resuming from checkpoints.

    Args:
        data_files (str or list): Path(s) to Parquet file(s).
        tokenizer (PreTrainedTokenizer): For the tokenization of text to token IDs.
        config (DictConfig): Options like cache_dir, prompt_key, max_prompt_length, truncation, etc.
        processor (ProcessorMixin, optional): Multimodal preprocessor for images/videos.
    """

    def __init__(
        self,
        data_files: str | list[str],
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        default_dir: str = "/path"
    ):
        if not isinstance(data_files, list | ListConfig):
            data_files = [data_files]
        self.data = []
        for data_file in data_files:
            with open(data_file) as f:
                for line in f.readlines():
                    self.data.append(line)
        self.processor = processor
        self.config = config
        self.max_prompt_length = config.get("max_prompt_length", 24576)
        self.max_response_length = config.get("max_response_length", 8192)
        self.remove_supplement_prompt = config.get("remove_supplement_prompt", False)
        self.truncation = config.get("truncation", "error")
        self.default_dir = default_dir
        prefix = """

Additional Output Requirements:"""
        suffix = "You need to generate content to ensure that such similar questions can be answered."
        self.pattern = re.escape(prefix) + r'.*?' + re.escape(suffix)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        row_dict = json.loads(self.data[item])
        if row_dict["type"] == "semantic":
            history_count = 4 if row_dict["step"] >= 4 else row_dict["step"]
            current = str(row_dict["video_idx"])
            history_semantic = [str(i) for i in range(int(current) - history_count, int(current))]
            video_id = row_dict["id"].split("*")[0]
            map_file = json.load(open(os.path.join(row_dict["episodic_folder"], video_id, f"{video_id}_map.json")))
            episodic_file = json.load(open(os.path.join(row_dict["episodic_folder"], video_id, f"episodic_{row_dict['memory_tag']}.json")))
            if row_dict["step"] > 0:
                semantic_file = json.load(open(os.path.join(self.default_dir, "history", f"{video_id}.json")))
            else:
                semantic_file = {}
            
            episodic_glo, semantic_glo = [], []
            # local2global
            for t in history_semantic:
                face_map  = map_file[t]
                episodic = episodic_file[t]
                semantic = semantic_file[t]
                try:
                    episodic = unify_face_id(episodic)
                except:
                    pass
                for glo, loc in face_map.items():
                    if loc in episodic:
                        episodic = episodic.replace(loc, glo)
                episodic_glo.append(episodic)
                for memory in semantic:
                    try:
                        memory = unify_face_id(memory)
                    except:
                        pass
                    for glo, loc in face_map.items():
                        if loc in memory:
                            memory = memory.replace(loc, glo)
                    semantic_glo.append(memory)
            
            # global2local
            face_map  = map_file[current]
            episodic_loc, semantic_loc = [], []
            for memory in episodic_glo:
                for glo, loc in face_map.items():
                    if glo in memory:
                        memory = memory.replace(glo, loc)
                episodic_loc.append(memory)
            for memory in semantic_glo:
                for glo, loc in face_map.items():
                    if glo in memory:
                        memory = memory.replace(glo, loc)
                semantic_loc.append(memory)
            
            # process unmapped global face
            pattern = r'\[face_\d+\]'
            str_under_process = "".join(episodic_loc + semantic_loc)
            result = set(re.findall(pattern, str_under_process))
            if len(result) > 0:
                loc_id, temp_map = len(face_map) + 1, {}
                for res in result:
                    temp_map[res] = f"<face_{loc_id}>"
                    loc_id += 1
                episodic_glo, semantic_glo = episodic_loc, semantic_loc
                episodic_loc, semantic_loc = [], []
                for memory in episodic_glo:
                    for glo, loc in temp_map.items():
                        if glo in memory:
                            memory = memory.replace(glo, loc)
                    episodic_loc.append(memory)
                for memory in semantic_glo:
                    for glo, loc in temp_map.items():
                        if glo in memory:
                            memory = memory.replace(glo, loc)
                    semantic_loc.append(memory)
            
            # append last episodic
            episodic_loc.append(episodic_file[current])

            row_dict["input"][-1]["text"] = row_dict["input"][-1]["text"].format(
                " ".join(episodic_loc),
                json.dumps(semantic_loc, indent=4, ensure_ascii=False)
            )
            row_dict = {
                "id": row_dict["id"],
                "type": "semantic",
                "input": row_dict["input"]
            }

        if row_dict["type"] == "episodic_on_policy":
            history_count = 4 if row_dict["step"] >= 4 else row_dict["step"]
            current = str(row_dict["video_idx"])
            video_id = row_dict["id"].split("*")[0]
            history_episodic = [str(i) for i in range(int(current) - history_count, int(current))]
            map_file = json.load(open(os.path.join(row_dict["episodic_folder"], video_id, f"{video_id}_map.json")))
            if row_dict["step"] > 0:
                episodic_file = json.load(open(os.path.join(self.default_dir, "history", f"{video_id}.json")))
            else:
                episodic_file = {}
            
            episodic_glo = []
            # local2global
            for t in history_episodic:
                face_map  = map_file[t]
                episodic = episodic_file[t]
                try:
                    episodic = unify_face_id(episodic)
                except:
                    pass
                for glo, loc in face_map.items():
                    if loc in episodic:
                        episodic = episodic.replace(loc, glo)
                episodic_glo.append(episodic)
            
            # global2local
            face_map  = map_file[current]
            episodic_loc = []
            for memory in episodic_glo:
                for glo, loc in face_map.items():
                    if glo in memory:
                        memory = memory.replace(glo, loc)
                episodic_loc.append(memory)
            
            # process unmapped global face
            pattern = r'\[face_\d+\]'
            str_under_process = "".join(episodic_loc)
            result = set(re.findall(pattern, str_under_process))
            if len(result) > 0:
                loc_id, temp_map = len(face_map) + 1, {}
                for res in result:
                    temp_map[res] = f"<face_{loc_id}>"
                    loc_id += 1
                episodic_glo = episodic_loc
                episodic_loc = []
                for memory in episodic_glo:
                    for glo, loc in temp_map.items():
                        if glo in memory:
                            memory = memory.replace(glo, loc)
                    episodic_loc.append(memory)

            row_dict["input"][-1]["text"] = row_dict["input"][-1]["text"].replace("{}", " ".join(episodic_loc))

            row_dict = {
                "id": row_dict["id"],
                "type": "episodic",
                "input": row_dict["input"]
            }

        if "text" in row_dict:
            text = row_dict["text"]
            images = row_dict["images"]
            videos = [tuple(torch.load(row_dict["videos"]))]
            video_kwargs = row_dict["video_kwargs"]
        else:
            messages = [{"role": "user", "content": row_dict["input"]}]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            images, videos, video_kwargs = process_vision_info(
                messages,
                image_patch_size=self.processor.image_processor.patch_size,
                return_video_kwargs=True,
                return_video_metadata=True
            )

        mm_data = {}
        if images is not None:
            mm_data['image'] = images
        if videos is not None:
            mm_data['video'] = videos
            
        videos, video_metadatas = zip(*videos)
        videos, video_metadatas = list(videos), list(video_metadatas)
        inputs = self.processor(text=text, images=images, videos=videos, video_metadata=video_metadatas, return_tensors="pt", **video_kwargs)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        row_dict["raw_prompt_ids"] = self.processor.tokenizer.encode(text, add_special_tokens=False)

        try:
            input_ids, attention_mask = verl_F.postprocess_data(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=self.max_prompt_length,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                left_pad=True,
                truncation=self.truncation,
            )
        except Exception as e:
            raise NotImplementedError(f"[{row_dict['id']}] sequence_length is larger than max_length")

        vision_position_ids = get_rope_index(
            self.processor,
            input_ids=input_ids[0],
            image_grid_thw=inputs.get("image_grid_thw"),
            video_grid_thw=inputs.get("video_grid_thw"),
            second_per_grid_ts=inputs.get("second_per_grid_ts"),
            attention_mask=attention_mask[0],
        )  # (3, seq_length)
        valid_mask = attention_mask[0].bool()
        text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
        text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
        position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]
        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        row_dict["multi_modal_inputs"] = {
            "pixel_values_videos": inputs["pixel_values_videos"],
            "video_grid_thw": inputs["video_grid_thw"],
        }
        row_dict["multi_modal_data"] = mm_data
        row_dict["mm_processor_kwargs"] = video_kwargs
        
        row_dict["tools_kwargs"] = {}
        row_dict["interaction_kwargs"] = {}
        return row_dict
    
    def modify_batch(self, batch):
        batch.batch["input_ids"][..., :self.max_prompt_length] = batch.batch.pop("bak_input_ids")
        batch.batch["attention_mask"][..., :self.max_prompt_length] = batch.batch.pop("bak_attention_mask")
        batch.batch["position_ids"][..., :self.max_prompt_length] = batch.batch.pop("bak_position_ids")
