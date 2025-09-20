"""
Remap the tokenized dataset from Llama3 to Qwen3.
"""

import json
import atexit
import multiprocessing
from pathlib import Path
from typing import Dict

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from streaming import LocalDataset, MDSWriter

min_length = 65536
src_root = Path("datasets/Meta-Llama-3/long-context-65536")
dst_root = Path("datasets/Qwen3/long-context-65536")

class Worker:

    def __init__(self):
        try:
            Worker.src_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", local_files_only=True)
        except Exception:
            Worker.src_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
        try:
            Worker.dst_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
        except Exception:
            Worker.dst_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

    def remap(data: Dict) -> Dict:
        indices, input_ids = data["indices"], data["input_ids"]
        input_ids_unpacked = [input_ids[indices[i][0]:indices[i][1]] for i in range(len(indices))]
        input_texts_unpacked = [Worker.src_tokenizer.decode(ids) for ids in input_ids_unpacked]
        input_ids_remapped = [Worker.dst_tokenizer.encode(text) for text in input_texts_unpacked]
        indices_remapped = []
        cur_pos = 0
        for ids in input_ids_remapped:
            indices_remapped.append((cur_pos, cur_pos + len(ids)))
            cur_pos += len(ids)
        input_ids_remapped_concat = []
        for ids in input_ids_remapped:
            input_ids_remapped_concat.extend(ids)
        indices = np.array(indices_remapped, dtype=np.uint32)
        input_ids = np.array(input_ids_remapped_concat, dtype=np.uint32)
        data["indices"], data["input_ids"] = indices, input_ids
        return data

pool = multiprocessing.Pool(processes=64, initializer=Worker)
atexit.register(pool.terminate)

for src_subset in src_root.iterdir():
    if src_subset.is_file():
        continue
    if src_subset.name.startswith("."):
        continue
    with Path(src_subset, "index.json").open("r") as f:
        index = json.load(f)
        shard = index["shards"][0]
        column_names = shard["column_names"]
        column_encodings = shard["column_encodings"]
        columns = {key: val for key, val in zip(column_names, column_encodings)}
    dst_subset = Path(dst_root, src_subset.name)
    with MDSWriter(columns=columns, out=dst_subset.as_posix()) as dst_writer:
        dataset = LocalDataset(src_subset)
        futures = pool.imap(Worker.remap, dataset, chunksize=32)
        for data in tqdm(futures, ncols=80, desc="Remapping %s" % src_subset.name, total=len(dataset), mininterval=30.0):
            dst_writer.write(data)
