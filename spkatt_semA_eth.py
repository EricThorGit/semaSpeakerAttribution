# %% [markdown]""
# # Setup
# 

# %% [markdown]
# ## GPUs

# %%
# set gpus for qlora training
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4, 5, 6, 7"

# %%
# device map for 7b model
device_map = {
    "model.embed_tokens": 0,
    "model.layers.0": 0,
    "model.layers.1": 0,
    "model.layers.2": 0,
    "model.layers.3": 0,
    "model.layers.4": 0,
    "model.layers.5": 0,
    "model.layers.6": 0,
    "model.layers.7": 0,
    "model.layers.8": 0,
    "model.layers.9": 0,
    "model.layers.10": 0,
    "model.layers.11": 0,
    "model.layers.12": 0,
    "model.layers.13": 0,
    "model.layers.14": 0,
    "model.layers.15": 0,
    "model.layers.16": 0,
    "model.layers.17": 0,
    "model.layers.18": 0,
    "model.layers.19": 0,
    "model.layers.20": 0,
    "model.layers.21": 0,
    "model.layers.22": 0,
    "model.layers.23": 0,
    "model.layers.24": 0,
    "model.layers.25": 0,
    "model.layers.26": 0,
    "model.layers.27": 0,
    "model.layers.28": 0,
    "model.layers.29": 0,
    "model.layers.30": 0,
    "model.layers.31": 0,
    "model.norm": 0,
    "lm_head": 0,
}

# device map for 70b model
# device_map = {
#     "model.embed_tokens": 0,
#     "model.layers.0": 0,
#     "model.layers.1": 0,
#     "model.layers.2": 0,
#     "model.layers.3": 0,
#     "model.layers.4": 0,
#     "model.layers.5": 0,
#     "model.layers.6": 0,
#     "model.layers.7": 0,
#     "model.layers.8": 0,
#     "model.layers.9": 0,
#     "model.layers.10": 0,
#     "model.layers.11": 0,
#     "model.layers.12": 0,
#     "model.layers.13": 0,
#     "model.layers.14": 0,
#     "model.layers.15": 0,
#     "model.layers.16": 0,
#     "model.layers.17": 0,
#     "model.layers.18": 1,
#     "model.layers.19": 1,
#     "model.layers.20": 1,
#     "model.layers.21": 1,
#     "model.layers.22": 1,
#     "model.layers.23": 1,
#     "model.layers.24": 1,
#     "model.layers.25": 1,
#     "model.layers.26": 1,
#     "model.layers.27": 1,
#     "model.layers.28": 1,
#     "model.layers.29": 1,
#     "model.layers.30": 1,
#     "model.layers.31": 1,
#     "model.layers.32": 1,
#     "model.layers.33": 1,
#     "model.layers.34": 1,
#     "model.layers.35": 1,
#     "model.layers.36": 1,
#     "model.layers.37": 1,
#     "model.layers.38": 1,
#     "model.layers.39": 2,
#     "model.layers.40": 2,
#     "model.layers.41": 2,
#     "model.layers.42": 2,
#     "model.layers.43": 2,
#     "model.layers.44": 2,
#     "model.layers.45": 2,
#     "model.layers.46": 2,
#     "model.layers.47": 2,
#     "model.layers.48": 2,
#     "model.layers.49": 2,
#     "model.layers.50": 2,
#     "model.layers.51": 2,
#     "model.layers.52": 2,
#     "model.layers.53": 2,
#     "model.layers.54": 2,
#     "model.layers.55": 2,
#     "model.layers.56": 2,
#     "model.layers.57": 2,
#     "model.layers.58": 2,
#     "model.layers.59": 2,
#     "model.layers.60": 3,
#     "model.layers.61": 3,
#     "model.layers.62": 3,
#     "model.layers.63": 3,
#     "model.layers.64": 3,
#     "model.layers.65": 3,
#     "model.layers.66": 3,
#     "model.layers.67": 3,
#     "model.layers.68": 3,
#     "model.layers.69": 3,
#     "model.layers.70": 3,
#     "model.layers.71": 3,
#     "model.layers.72": 3,
#     "model.layers.73": 3,
#     "model.layers.74": 3,
#     "model.layers.75": 3,
#     "model.layers.76": 3,
#     "model.layers.77": 3,
#     "model.layers.78": 3,
#     "model.layers.79": 3,
#     "model.norm": 3,
#     "lm_head": 3,
# }

# %% [markdown]
# ## Imports
# 

# %%
import numpy as np
from tqdm import tqdm
import json
import warnings
import gc
import torch
import json
import Levenshtein
import shutil

from datasets import (
    load_dataset,
    concatenate_datasets,
    load_from_disk,
    Features,
    Sequence,
    Value,
)
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM, LlamaTokenizer
from langchain import HuggingFacePipeline, PromptTemplate, LLMChain
from peft import PeftModel

from datasets import logging as ds_logging
from transformers import logging as trans_logging

from qlora import train


# %% [markdown]
# ## Logging
# 

# %%
ds_logging.set_verbosity_error()
ds_logging.disable_progress_bar()
trans_logging.set_verbosity_error()
warnings.filterwarnings("ignore")

# %% [markdown]
# # Data
# 

# %% [markdown]
# ## Load datasets
# 

# %%
def read_annotations_from_file(path: str, file: str):
    features = Features(
        {
            "PTC": Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
            "Evidence": Sequence(
                feature=Value(dtype="string", id=None), length=-1, id=None
            ),
            "Medium": Sequence(
                feature=Value(dtype="string", id=None), length=-1, id=None
            ),
            "Topic": Sequence(
                feature=Value(dtype="string", id=None), length=-1, id=None
            ),
            "Cue": Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
            "Addr": Sequence(
                feature=Value(dtype="string", id=None), length=-1, id=None
            ),
            "Message": Sequence(
                feature=Value(dtype="string", id=None), length=-1, id=None
            ),
            "Source": Sequence(
                feature=Value(dtype="string", id=None), length=-1, id=None
            ),
        }
    )
    ds = load_dataset(
        "json",
        data_files=os.path.join(path, file),
        field="Annotations",
        split="train",
        features=features,
    )
    ds = ds.add_column("FileName", [file] * len(ds))
    return ds

# %%
def read_sentences_from_file(path: str, file: str):
    ds = load_dataset(
        "json", data_files=os.path.join(path, file), field="Sentences", split="train"
    )
    ds = ds.add_column("FileName", [file] * len(ds))
    ds = ds.add_column("Sentence", [" ".join(t) for t in ds["Tokens"]])
    return ds

# %%
def read_annotations_from_path(path: str):
    dataset = None

    for file in tqdm(sorted(os.listdir(path))):
        if not dataset:
            dataset = read_annotations_from_file(path, file)
        else:
            dataset = concatenate_datasets(
                [dataset, read_annotations_from_file(path, file)]
            )

    return dataset

# %%
def read_sentences_from_path(path: str):
    dataset = None

    for file in tqdm(sorted(os.listdir(path))):
        if not dataset:
            dataset = read_sentences_from_file(path, file)
        else:
            dataset = concatenate_datasets(
                [dataset, read_sentences_from_file(path, file)]
            )

    dataset = dataset.add_column("id", range(len(dataset)))
    return dataset

# %%
def read_sentences_dataset(ds_name: str):
    path_to_dataset = "./transformed_datasets/" + ds_name + "/sentences"

    if os.path.isdir(path_to_dataset):
        result = load_from_disk(path_to_dataset)
    else:
        result = read_sentences_from_path(
            "./SpkAtt-2023/data/"
            + ds_name
            + "/task1"
            + ("_test/" if ds_name == "eval" else "/")
        )
        os.makedirs(path_to_dataset, exist_ok=True)
        result.save_to_disk(path_to_dataset)

    return result

# %%
def read_annotations_dataset(ds_name: str):
    path_to_dataset = "./transformed_datasets/" + ds_name + "/annotations"

    if os.path.isdir(path_to_dataset):
        return load_from_disk(path_to_dataset)

    result = read_annotations_from_path(
        "./SpkAtt-2023/data/"
        + ds_name
        + "/task1"
        + ("_test/" if ds_name == "eval" else "/")
    )
    os.makedirs(path_to_dataset, exist_ok=True)
    result.save_to_disk(path_to_dataset)
    return result

# %%
train_sentences_dataset = read_sentences_dataset("train")
val_sentences_dataset = read_sentences_dataset("dev")
test_sentences_dataset = read_sentences_dataset("eval")

# %%
train_annotations_dataset = read_annotations_dataset("train")
val_annotations_dataset = read_annotations_dataset("dev")

# %% [markdown]
# ## Format datasets for usage in langchain
# 

# %%
def get_text_from_label(train_sentences_dataset, row, annotations):
    tokens = []
    for anno in annotations:
        if int(anno.split(":")[0]) == row["SentenceId"]:
            tokens.append(row["Tokens"][int(anno.split(":")[1])])
        else:
            temp_row = train_sentences_dataset.filter(
                lambda r: r["FileName"] == row["FileName"]
                and r["SentenceId"] == int(anno.split(":")[0])
            )[0]
            tokens.append(temp_row["Tokens"][int(anno.split(":")[1])])
    return tokens

# %%
def build_complete_dataset(sentences_dataset, annotations_dataset, dataset_name):
    path_to_dataset = "./transformed_datasets/" + dataset_name + "/complete"
    if os.path.isdir(path_to_dataset):
        return load_from_disk(path_to_dataset)

    ptc, ptc_temp, ptc_mapped, ptc_mapped_temp = [], [], [], []
    evidence, evidence_temp, evidence_mapped, evidence_mapped_temp = [], [], [], []
    medium, medium_temp, medium_mapped, medium_mapped_temp = [], [], [], []
    topic, topic_temp, topic_mapped, topic_mapped_temp = [], [], [], []
    cue, cue_temp, cue_mapped, cue_mapped_temp = [], [], [], []
    addr, addr_temp, addr_mapped, addr_mapped_temp = [], [], [], []
    message, message_temp, message_mapped, message_mapped_temp = [], [], [], []
    source, source_temp, source_mapped, source_mapped_temp = [], [], [], []
    (
        sentence_extended,
        tokens_extended,
        sentence_extended_ids,
    ) = (
        [],
        [],
        [],
    )

    index_in_anno_ds = 0

    for i, row in tqdm(enumerate(sentences_dataset)):
        context = row["Sentence"]
        tokens = row["Tokens"]
        ids = [row["SentenceId"]] * len(row["Tokens"])
        if (
            i + 1 < len(sentences_dataset)
            and sentences_dataset[i + 1]["FileName"] == row["FileName"]
        ):
            context = context + " " + sentences_dataset[i + 1]["Sentence"]
            tokens.extend(sentences_dataset[i + 1]["Tokens"])
            ids.extend(
                [sentences_dataset[i + 1]["SentenceId"]]
                * len(sentences_dataset[i + 1]["Tokens"])
            )
        if (
            i + 2 < len(sentences_dataset)
            and sentences_dataset[i + 2]["FileName"] == row["FileName"]
        ):
            context = context + " " + sentences_dataset[i + 2]["Sentence"]
            tokens.extend(sentences_dataset[i + 2]["Tokens"])
            ids.extend(
                [sentences_dataset[i + 2]["SentenceId"]]
                * len(sentences_dataset[i + 2]["Tokens"])
            )
        sentence_extended.append(context)
        tokens_extended.append(tokens)
        sentence_extended_ids.append(ids)

        if annotations_dataset is not None:
            id_of_next_sentence_with_annotation = (
                int(annotations_dataset[index_in_anno_ds]["Cue"][0].split(":")[0])
                if index_in_anno_ds != len(annotations_dataset)
                else -1
            )

            if row["SentenceId"] != id_of_next_sentence_with_annotation:
                ptc.append([])
                ptc_mapped.append([])
                evidence.append([])
                evidence_mapped.append([])
                medium.append([])
                medium_mapped.append([])
                topic.append([])
                topic_mapped.append([])
                cue.append([])
                cue_mapped.append([])
                addr.append([])
                addr_mapped.append([])
                message.append([])
                message_mapped.append([])
                source.append([])
                source_mapped.append([])
                continue

            while row["SentenceId"] == id_of_next_sentence_with_annotation:
                ptc_temp.append(annotations_dataset[index_in_anno_ds]["PTC"])
                evidence_temp.append(annotations_dataset[index_in_anno_ds]["Evidence"])
                medium_temp.append(annotations_dataset[index_in_anno_ds]["Medium"])
                topic_temp.append(annotations_dataset[index_in_anno_ds]["Topic"])
                cue_temp.append(annotations_dataset[index_in_anno_ds]["Cue"])
                addr_temp.append(annotations_dataset[index_in_anno_ds]["Addr"])
                message_temp.append(annotations_dataset[index_in_anno_ds]["Message"])
                source_temp.append(annotations_dataset[index_in_anno_ds]["Source"])

                ptc_mapped_temp.append(
                    get_text_from_label(sentences_dataset, row, ptc_temp[-1])
                )
                evidence_mapped_temp.append(
                    get_text_from_label(sentences_dataset, row, evidence_temp[-1])
                )
                medium_mapped_temp.append(
                    get_text_from_label(sentences_dataset, row, medium_temp[-1])
                )
                topic_mapped_temp.append(
                    get_text_from_label(sentences_dataset, row, topic_temp[-1])
                )
                cue_mapped_temp.append(
                    get_text_from_label(sentences_dataset, row, cue_temp[-1])
                )
                addr_mapped_temp.append(
                    get_text_from_label(sentences_dataset, row, addr_temp[-1])
                )
                message_mapped_temp.append(
                    get_text_from_label(sentences_dataset, row, message_temp[-1])
                )
                source_mapped_temp.append(
                    get_text_from_label(sentences_dataset, row, source_temp[-1])
                )

                index_in_anno_ds += 1
                if index_in_anno_ds == len(annotations_dataset):
                    break
                id_of_next_sentence_with_annotation = int(
                    annotations_dataset[index_in_anno_ds]["Cue"][0].split(":")[0]
                )

            ptc.append(ptc_temp)
            ptc_mapped.append(ptc_mapped_temp)
            evidence.append(evidence_temp)
            evidence_mapped.append(evidence_mapped_temp)
            medium.append(medium_temp)
            medium_mapped.append(medium_mapped_temp)
            topic.append(topic_temp)
            topic_mapped.append(topic_mapped_temp)
            cue.append(cue_temp)
            cue_mapped.append(cue_mapped_temp)
            addr.append(addr_temp)
            addr_mapped.append(addr_mapped_temp)
            message.append(message_temp)
            message_mapped.append(message_mapped_temp)
            source.append(source_temp)
            source_mapped.append(source_mapped_temp)

            ptc_temp, ptc_mapped_temp = [], []
            evidence_temp, evidence_mapped_temp = [], []
            medium_temp, medium_mapped_temp = [], []
            topic_temp, topic_mapped_temp = [], []
            cue_temp, cue_mapped_temp = [], []
            addr_temp, addr_mapped_temp = [], []
            message_temp, message_mapped_temp = [], []
            source_temp, source_mapped_temp = [], []

    res = sentences_dataset.add_column("sentence_extended", sentence_extended)
    res = res.add_column("tokens_extended", tokens_extended)
    res = res.add_column("sentence_extended_ids", sentence_extended_ids)

    if annotations_dataset is not None:
        res = res.add_column("ptc", ptc)
        res = res.add_column("ptc_mapped", ptc_mapped)
        res = res.add_column("evidence", evidence)
        res = res.add_column("evidence_mapped", evidence_mapped)
        res = res.add_column("medium", medium)
        res = res.add_column("medium_mapped", medium_mapped)
        res = res.add_column("topic", topic)
        res = res.add_column("topic_mapped", topic_mapped)
        res = res.add_column("cue", cue)
        res = res.add_column("cue_mapped", cue_mapped)
        res = res.add_column("addr", addr)
        res = res.add_column("addr_mapped", addr_mapped)
        res = res.add_column("message", message)
        res = res.add_column("message_mapped", message_mapped)
        res = res.add_column("source", source)
        res = res.add_column("source_mapped", source_mapped)

    os.makedirs(path_to_dataset, exist_ok=True)
    res.save_to_disk(path_to_dataset)

    return res

# %%
train_ds = build_complete_dataset(
    train_sentences_dataset, train_annotations_dataset, "train"
)
val_ds = build_complete_dataset(val_sentences_dataset, val_annotations_dataset, "dev") # USE FOR INFERENCE
test_ds = build_complete_dataset(test_sentences_dataset, None, "eval")

# %%
inputs = test_sentences_dataset.rename_column("Sentence", "Satz")

# %% [markdown]
# ## Dataset Showcase
# 

# %%
train_ds[52]

# %%
train_ds[15]

# %% [markdown]
# ## Build lmsys format json
# 

# %%
def map_cues_to_string(mapped):
    if mapped == []:
        return "#UNK#"
    return ", ".join(["[" + ", ".join(val) + "]" for val in mapped])

# %%
def map_roles_to_string(mapped):
    if mapped == []:
        return "#UNK#"
    return ", ".join(mapped)

# %%
lmsys_data_path = "./lmsys.json"


def build_lmsys_format(train_ds, val_ds):
    result = []

    index = 0
    for row in train_ds: # CHANGED 
        if len(row["cue_mapped"]) == 0:
            element = {"id": "identity_" + str(index)}
            index += 1
            conversations = [
                {
                    "from": "human",
                    "value": 'A cue is the lexical items in a sentence that indicate that speech, writing, or thought is being reproduced.\nI want you to extract all cues in the text below.\nIf you find multiple words for one cue, you output them separated by commas.\nIf no cue can be found in the given text, you output the string #UNK# as cue.\nNow extract all cues from the following sentence.\nUse the prefix "Cues: ".\nSentence: '
                    + row["Sentence"],
                },
                {
                    "from": "gpt",
                    "value": "Cues: " + map_cues_to_string(row["cue_mapped"]),
                },
            ]
            element["conversations"] = conversations
            result.append(element)
            continue
        for i, cue in enumerate(row["cue_mapped"]):
            element = {"id": "identity_" + str(index)}
            index += 1
            conversations = [
                {
                    "from": "human",
                    "value": 'A cue is the lexical items in a sentence that indicate that speech, writing, or thought is being reproduced.\nI want you to extract all cues in the text below.\nIf you find multiple words for one cue, you output them separated by commas.\nIf no cue can be found in the given text, you output the string #UNK# as cue.\nNow extract all cues from the following sentence.\nUse the prefix "Cues: ".\nSentence: '
                    + row["Sentence"],
                },
                {
                    "from": "gpt",
                    "value": "Cues: " + map_cues_to_string(row["cue_mapped"]),
                },
                {
                    "from": "human",
                    "value": "Now I give you again the sentence only in addition with the two following sentences, because the roles can be partially contained in the following sentences.\nText: "
                    + row["sentence_extended"]
                    + "\n\nNow find all roles in the sentence associated with the cue '"
                    + ", ".join(cue)
                    + "' you found in the beginning sentence.",
                },
                {
                    "from": "gpt",
                    "value": "cue: "
                    + ", ".join(cue)
                    + "\nptc: "
                    + map_roles_to_string(row["ptc_mapped"][i])
                    + "\nevidence: "
                    + map_roles_to_string(row["evidence_mapped"][i])
                    + "\nmedium: "
                    + map_roles_to_string(row["medium_mapped"][i])
                    + "\ntopic: "
                    + map_roles_to_string(row["topic_mapped"][i])
                    + "\naddr: "
                    + map_roles_to_string(row["addr_mapped"][i])
                    + "\nmessage: "
                    + map_roles_to_string(row["message_mapped"][i])
                    + "\nsource: "
                    + map_roles_to_string(row["source_mapped"][i]),
                },
            ]
            element["conversations"] = conversations
            result.append(element)

    with open(lmsys_data_path, "w", encoding="utf8") as outfile:
        json.dump(result, outfile, indent=3)

# %%
build_lmsys_format(train_ds, val_ds)

# %% [markdown]
# # QLoRA Fine-Tuning
# 
# ## Parse data into required format
# 

# %%
parsed_cues_file = "./transformed_datasets/prompts_training/parsed_data_cues.jsonl"
parsed_roles_file = "./transformed_datasets/prompts_training/parsed_data_roles.jsonl"
os.makedirs(os.path.dirname(parsed_cues_file), exist_ok=True)
os.makedirs(os.path.dirname(parsed_roles_file), exist_ok=True)

# token to signal the end of the assistant's response
separator = "</s>"

# reload parsed data
with open(lmsys_data_path) as f:
    data = json.load(f)

# save parsed prompts separately
all_prompts_cues = []
all_prompts_roles = []
for conversation in data:
    # keep track of the complete conversation in order to generate the input of the prompts
    complete_prompt = ""

    for i, turn in enumerate(conversation["conversations"]):
        if turn["from"] == "human":
            complete_prompt += "User: "
            complete_prompt += turn["value"]
        elif turn["from"] == "gpt":
            complete_prompt += "Assistant: "

            # idea
            # turn 0: user prompt for cues
            # turn 1: assistant response with cues
            #   --> create sample with the conversation up to this point as input and the cues as output
            # turn 2: user prompt for roles for one specific cue
            # turn 3: assistant response with roles
            #   --> create sample with the conversation up to this point as input and the roles as output
            # there should be no further turns because we split all conversations with multiple cues into separate conversations

            sample = json.dumps(
                {"input": complete_prompt, "output": turn["value"] + separator}
            )

            if i == 1 and sample not in all_prompts_cues:
                # turn 1: assistant response with cues
                all_prompts_cues.append(sample)
            elif i == 3 and sample not in all_prompts_cues:
                # turn 3: assistant response with roles
                all_prompts_roles.append(sample)
            elif i != 1 and i != 3:
                print(
                    "ERROR: each conversation should maximally contain 4 turns"
                    " and only turn 1 and 3 should be responses by the assistant"
                )

            complete_prompt += turn["value"] + separator
        complete_prompt += "\n"

# write parsed prompts to files
with open(parsed_cues_file, "w") as f:
    f.write("\n".join(all_prompts_cues))

with open(parsed_roles_file, "w") as f:
    f.write("\n".join(all_prompts_roles))

# %%
# check that the file with the cue prompts was written correctly
with open(parsed_cues_file) as f:
    lines = f.readlines()

print(f"Number of samples: {len(lines)}\n")

print("First 5 samples:")
for l in lines[:5]:
    print("=== in: ===\n" + json.loads(l)["input"] + "\n")
    print("=== out: ===\n" + json.loads(l)["output"] + "\n")
    print()

# %%
# check that the file with the role prompts was written correctly
with open(parsed_roles_file) as f:
    lines = f.readlines()

print(f"Number of samples: {len(lines)}\n")

print("First 5 samples:")
for l in lines[:5]:
    print("=== in: ===\n" + json.loads(l)["input"] + "\n")
    print("=== out: ===\n" + json.loads(l)["output"] + "\n")
    print()

# %% [markdown]
# ## Check optimal source and target lengths
# 
# This step is only required if you want to use your own data. If you use the original GermEval 2023 task 1 data, you can skip this step and use the source and target lengths that are already defined in the configurations below at the start of the training code (parameters `source_max_len` and `target_max_len`).
# 
# If you want to change the maximum source or target lengths, keep in mind that longer prompts mean longer training times and more memory requirements. While it would be best to set the maximum source/target lengths to the maximum lengths of the inputs/outputs, this is not always feasible due to memory constraints. In this case, we recommend choosing maximum lengths that only truncate few samples.
# 

# %%
# encode all prompt inputs with the Llama 1 tokenizer (same as the Llama 2 tokenizer)
tokenizer = AutoTokenizer.from_pretrained(
    "huggyllama/llama-7b", padding_side="right", use_fast=False, tokenizer_type="llama"
)

encoded_inputs_cues = []
encoded_inputs_roles = []
encoded_outputs_cues = []
encoded_outputs_roles = []
with open(parsed_cues_file) as f:
    for l in f.readlines():
        enc_in = tokenizer.encode(json.loads(l)["input"])
        encoded_inputs_cues.append(enc_in)
        enc_out = tokenizer.encode(json.loads(l)["output"])
        encoded_outputs_cues.append(enc_out)
with open(parsed_roles_file) as f:
    for l in f.readlines():
        enc_in = tokenizer.encode(json.loads(l)["input"])
        encoded_inputs_roles.append(enc_in)
        enc_out = tokenizer.encode(json.loads(l)["output"])
        encoded_outputs_roles.append(enc_out)

# %%
# maximum source lengths taken from the config files
max_length_source_cues = 256
max_length_source_roles = 640

print("cues source lengths")
len_enc = [len(e) for e in encoded_inputs_cues]
print(f"max length: {max(len_enc)}")
print(f"mean length: {np.mean(len_enc)}")
print(
    f"number of samples longer than {max_length_source_cues}: {sum(np.array(len_enc) > max_length_source_cues)}"
)
print()

print("roles source lengths")
len_enc = [len(e) for e in encoded_inputs_roles]
print(f"max length: {max(len_enc)}")
print(f"mean length: {np.mean(len_enc)}")
print(
    f"number of samples longer than {max_length_source_roles}: {sum(np.array(len_enc) > max_length_source_roles)}"
)

# %%
# maximum target lengths taken from the config files
max_length_target_cues = 64
max_length_target_roles = 256

print("cues target lengths")
len_enc = [len(e) for e in encoded_outputs_cues]
print(f"max length: {max(len_enc)}")
print(f"mean length: {np.mean(len_enc)}")
print(
    f"number of samples longer than {max_length_target_cues}: {sum(np.array(len_enc) > max_length_target_cues)}"
)
print()

print("roles target lengths")
len_enc = [len(e) for e in encoded_outputs_roles]
print(f"max length: {max(len_enc)}")
print(f"mean length: {np.mean(len_enc)}")
print(
    f"number of samples longer than {max_length_target_roles}: {sum(np.array(len_enc) > max_length_target_roles)}"
)

# %% [markdown]
# ## Train models
# 
# This step can be skipped if you already have trained models.
# 
# For training, you first have to prepare the Llama 2 models and adapt the configuration. To prepare the Llama 2 models, you will have to make them accessible in HF (Huggingface) format. You can either use the models directly from Huggingface or prepare them yourself by first downloading the model weights from [the official Llama repo](https://github.com/facebookresearch/llama) and then converting these weights using their [conversion manual](https://github.com/facebookresearch/llama-recipes/#model-conversion-to-hugging-face). When using the models from Huggingface, you should add the parameter `use_auth_token` with your Huggingface token to the training configs in the code cell below. If you don't want to use the models from Huggingface, once you have prepared the models yourself, update the path to the models in the config (parameter `model_name_or_path`) so the paths point to the folder containing the `pytorch_model-000xx-of-00015.bin` files.
# 
# Further configuration parameters:
# 
# - `per_device_train_batch_size` and `gradient_accumulation_steps`: With these two parameters you can control the batch size and the number of accumulation steps when calculating the gradients during training. Larger batch sizes should speed up training, but increase memory requirements considerably. We recommend choosing the parameters so that their product `per_device_train_batch_size * gradient_accumulation_steps` is a multiple of 16.
# - `save_steps` and `max_steps`: set `max_steps` to control the length of training (`save_steps` determines when checkpoints are created)
# 

# %%
# define config files for training
# 7B models
cues_training_config = {
    "model_name_or_path": "LeoLM/leo-hessianai-7b",
    "output_dir": "./output/spkatt-7b-cues-leolm",
    "data_seed": 42,
    "save_steps": 500,
    "evaluation_strategy": "no",
    "dataloader_num_workers": 4,
    "lora_modules": "all",
    "bf16": True,
    "dataset": "transformed_datasets/prompts_training/parsed_data_cues.jsonl",
    "dataset_format": "input-output",
    "source_max_len": 256,
    "target_max_len": 64,
    "per_device_train_batch_size": 16,
    "gradient_accumulation_steps": 1,
    "max_steps": 2000,
    "learning_rate": 0.0002,
    "lora_dropout": 0.1,
    "seed": 0,
}
roles_training_config = {
    "model_name_or_path": "LeoLM/leo-hessianai-7b",
    "output_dir": "./output/spkatt-7b-roles-leolm",
    "data_seed": 42,
    "save_steps": 500,
    "evaluation_strategy": "no",
    "dataloader_num_workers": 4,
    "lora_modules": "all",
    "bf16": True,
    "dataset": "transformed_datasets/prompts_training/parsed_data_roles.jsonl",
    "dataset_format": "input-output",
    "source_max_len": 640,
    "target_max_len": 256,
    "per_device_train_batch_size": 16,
    "gradient_accumulation_steps": 1,
    "max_steps": 2000,
    "learning_rate": 0.0002,
    "lora_dropout": 0.1,
    "seed": 0,
}

# 70B models
# cues_training_config = {"model_name_or_path": "meta-llama/Llama-2-70b-hf",
#                         "output_dir": "./output/spkatt-70b-cues",
#                         "data_seed": 42,
#                         "save_steps": 500,
#                         "evaluation_strategy": "no",
#                         "dataloader_num_workers": 4,
#                         "lora_modules": "all",
#                         "bf16": True,
#                         "dataset": "transformed_datasets/prompts_training/parsed_data_cues.jsonl",
#                         "dataset_format": "input-output",
#                         "source_max_len": 256,
#                         "target_max_len": 64,
#                         "per_device_train_batch_size": 16,
#                         "gradient_accumulation_steps": 1,
#                         "max_steps": 2000,
#                         "learning_rate": 0.0001,
#                         "lora_dropout": 0.05,
#                         "seed": 0,
#                         }
# roles_training_config = {"model_name_or_path": "meta-llama/Llama-2-70b-hf",
#                          "output_dir": "./output/spkatt-70b-roles",
#                          "data_seed": 42,
#                          "save_steps": 500,
#                          "evaluation_strategy": "no",
#                          "dataloader_num_workers": 4,
#                          "lora_modules": "all",
#                          "bf16": True,
#                          "dataset": "transformed_datasets/prompts_training/parsed_data_roles.jsonl",
#                          "dataset_format": "input-output",
#                          "source_max_len": 640,
#                          "target_max_len": 256,
#                          "per_device_train_batch_size": 8,
#                          "gradient_accumulation_steps": 2,
#                          "max_steps": 2500,
#                          "learning_rate": 0.0001,
#                          "lora_dropout": 0.05,
#                          "seed": 0,
#                          }

# %%
train(cues_training_config)

# free vram after training
gc.collect()
torch.cuda.empty_cache()
gc.collect()


# %%
train(roles_training_config)

# free vram after training
gc.collect()
torch.cuda.empty_cache()
gc.collect()


# %% [markdown]
# # Inference for Cues

# %% [markdown]
# ## Load cue model for inference

# %%
model = AutoModelForCausalLM.from_pretrained(
    cues_training_config["model_name_or_path"],
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

checkpoint_dir = (
    cues_training_config["output_dir"] + "/checkpoint-2000/" # 2000
)  # choose checkpoint
model = PeftModel.from_pretrained(model, os.path.join(checkpoint_dir, "adapter_model"))
model = model.merge_and_unload()

tokenizer = LlamaTokenizer.from_pretrained(
    cues_training_config["model_name_or_path"], legacy=False
)
tokenizer.bos_token_id = 1

pipe = pipeline(
    task="text-generation", model=model, tokenizer=tokenizer, max_new_tokens=300
)

llm = HuggingFacePipeline(pipeline=pipe)

# %% [markdown]
# ## Build Cue-LLM-Chain

# %%
template_cues = """User: A cue is the lexical items in a sentence that indicate that speech, writing, or thought is being reproduced.
I want you to extract all cues in the text below.
If you find multiple words for one cue, you output them separated by commas.
If no cue can be found in the given text, you output the string #UNK# as cue.
Now extract all cues from the following sentence.
Use the prefix \"Cues: \".
Sentence: {Satz}
Assistant:"""

# %%
prompt_cues = PromptTemplate(input_variables=["Satz"], template=template_cues)
llm_chain_cues = LLMChain(prompt=prompt_cues, llm=llm)

# %% [markdown]
# ## Inference

# %%
outputs_cues = []
for row in tqdm(inputs, desc="Cues"):
    outputs_cues.append(llm_chain_cues.apply([row])[0])

# %%
# free vram after inference for cues
del tokenizer
del model
del pipe
del llm
del llm_chain_cues
gc.collect()
torch.cuda.empty_cache()
gc.collect()
torch.cuda.empty_cache()
gc.collect()

# %% [markdown]
# # Save and extract cues

# %% [markdown]
# ## Save data in correct format and insert raw cue outputs

# %%
def map_outputs_to_output_file_format(inputs, outputs):
    result = {}
    seen_sentences = []

    for i, row in enumerate(inputs):
        if row["FileName"] not in result:
            result[row["FileName"]] = {
                "Sentences": [],
                "Annotations": [],
                "Outputs": {"Cues": {}},
            }

        if row["FileName"] + "-" + str(row["SentenceId"]) not in seen_sentences:
            seen_sentences.append(row["FileName"] + "-" + str(row["SentenceId"]))
            result[row["FileName"]]["Sentences"].append(
                {"SentenceId": row["SentenceId"], "Tokens": row["Tokens"]}
            )

        if row["SentenceId"] not in result[row["FileName"]]["Outputs"]["Cues"]:
            result[row["FileName"]]["Outputs"]["Cues"][row["SentenceId"]] = []

        result[row["FileName"]]["Outputs"]["Cues"][row["SentenceId"]].append(
            outputs[i]["text"]
        )

    return result


# %%
def save_outputs_to_output_files(inputs, outputs):
    path = "./output/data/"
    os.makedirs(path, exist_ok=True)
    for key, value in map_outputs_to_output_file_format(inputs, outputs).items():
        with open(path + key, "w", encoding="utf8") as outfile:
            json.dump(value, outfile, indent=3)


# %%
save_outputs_to_output_files(inputs, outputs_cues)

# %% [markdown]
# ## Map raw cue outputs to cues

# %%
def check_for_overlap(cues):
    for i, cue in enumerate(cues):
        for j in range(i + 1, len(cues)):
            if len(list(set(cue) & set(cues[j]))) > 0:
                return True, i, j
    return False, -1, -1


# %%
def extract_cues_from_output(output_string: str):
    output_string = output_string.strip().split("\n")[0].strip()

    if output_string.startswith("Cues:"):
        output_string = output_string[5:].strip()
    else:
        raise SystemError

    if output_string == "" or output_string == "#UNK#":
        return []

    outputs = [v.strip() for v in output_string.strip().split("],")]

    cues = []
    for i, output in enumerate(outputs):
        if i < len(outputs) - 1:
            output = output + "]"
        if not output.startswith("[") or not output.endswith("]"):
            raise LookupError
        output = output[1:-1]
        output = [v.strip().split(" ")[0].strip() for v in output.strip().split(",")]

        while "#UNK#" in output:
            output.pop(output.index("#UNK#"))

        cues.append(output)

    overlap, i, j = check_for_overlap(cues)
    while overlap:
        cue_2 = cues.pop(j)
        cue_1 = cues.pop(i)
        cue_1.extend(cue_2)
        cue_1 = list(set(cue_1))
        cues.append(cue_1)

        overlap, i, j = check_for_overlap(cues)

    return cues


# %%
def extract_cues():
    path = "./output/data/"
    count_cues = 0

    for file in sorted(os.listdir(path)):
        if file.endswith(".zip"):
            continue
        file_content = {}

        with open(os.path.join(path, file), "r") as f:
            file_content = json.load(f)
            file_content["Outputs"]["Cues_text"] = {}

            for id, output in file_content["Outputs"]["Cues"].items():
                try:
                    cues = extract_cues_from_output(output[0])
                # output does not start with "Cues: "
                except SystemError:
                    cues = []
                # output not in [...] format
                except LookupError:
                    cues = []

                count_cues += len(cues)
                file_content["Outputs"]["Cues_text"][id] = cues

        with open(os.path.join(path, file), "w", encoding="utf8") as outfile:
            json.dump(file_content, outfile, indent=3)

    return count_cues

# %%
count_cues = extract_cues()


# %% [markdown]
# # Inference for Roles

# %% [markdown]
# ## Load roles model for inference

# %%
model = AutoModelForCausalLM.from_pretrained(
    roles_training_config["model_name_or_path"],
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

checkpoint_dir = (
    roles_training_config["output_dir"] + "/checkpoint-2000/"
)  # choose checkpoint
model = PeftModel.from_pretrained(model, os.path.join(checkpoint_dir, "adapter_model"))
model = model.merge_and_unload()

tokenizer = LlamaTokenizer.from_pretrained(
    roles_training_config["model_name_or_path"], legacy=False
)
tokenizer.bos_token_id = 1

pipe = pipeline(
    task="text-generation", model=model, tokenizer=tokenizer, max_new_tokens=300
)

llm = HuggingFacePipeline(pipeline=pipe)

# %% [markdown]
# ## Inference

# %%
def prompt_for_roles(ds, prompt_cues, roles_prompt):
    path = "./output/data/"

    pbar = tqdm(total=count_cues, desc="Roles")
    for file in sorted(os.listdir(path)):
        if file.endswith(".zip"):
            continue
        file_content = {}

        with open(os.path.join(path, file), "r") as f:
            file_content = json.load(f)
            file_content["Outputs"]["Roles"] = {}

            for id, cues in file_content["Outputs"]["Cues_text"].items():
                file_content["Outputs"]["Roles"][id] = []

                if cues == []:
                    continue

                sentence = ds.filter(
                    lambda r: r["FileName"] == file and r["SentenceId"] == int(id)
                )[0]["Sentence"]
                text = ds.filter(
                    lambda r: r["FileName"] == file and r["SentenceId"] == int(id)
                )[0]["sentence_extended"]
                if sentence.endswith(":"):
                    sentence = sentence[:-1] + "."
                if text.endswith(":"):
                    text = text[:-1] + "."
                cue_prompt = (
                    prompt_cues.format(Satz=sentence)
                    + " Cues: "
                    + ", ".join(["[" + ", ".join(cue) + "]" for cue in cues])
                    + "</s>"
                )

                for cue in cues:
                    file_content["Outputs"]["Roles"][id].append([])
                    prompt = PromptTemplate(
                        input_variables=["text", "cue"],
                        template=cue_prompt + "\nUser: " + roles_prompt,
                    )
                    llm_chain = LLMChain(prompt=prompt, llm=llm)
                    output = llm_chain.apply([{"text": text, "cue": ", ".join(cue)}])[
                        0
                    ]["text"]
                    file_content["Outputs"]["Roles"][id][-1].append(output)
                    pbar.update()

        with open(os.path.join(path, file), "w", encoding="utf8") as outfile:
            json.dump(file_content, outfile, indent=3)

    pbar.close()


# %%
roles_prompt = "Now I give you again the sentence only in addition with the two following sentences, because the roles can be partially contained in the following sentences.\nText: {text}\n\nNow find all roles in the sentence associated with the cue '{cue}' you found in the beginning sentence.\nAssistant:"


# %%
prompt_for_roles(test_ds, prompt_cues, roles_prompt)

# %%
# free vram after inference for roles
del tokenizer
del model
del pipe
del llm
gc.collect()
torch.cuda.empty_cache()
gc.collect()
torch.cuda.empty_cache()
gc.collect()


# %% [markdown]
# # Extract Roles and map outputs to tokens

# %% [markdown]
# ## Extract roles

# %%
def extract_roles_from_output(output_string: str):
    res = {
        "ptc": "",
        "evidence": "",
        "medium": "",
        "topic": "",
        "addr": "",
        "message": "",
        "source": "",
    }

    output_rows = [v.strip() for v in output_string.strip().split("\n")]

    try:
        if output_rows[1].startswith("ptc: "):
            res["ptc"] = [
                v.strip().split(" ")[0].strip()
                for v in output_rows[1][4:].strip().split(",")
            ]
    except IndexError:
        pass
    try:
        if output_rows[2].startswith("evidence: "):
            res["evidence"] = [
                v.strip().split(" ")[0].strip()
                for v in output_rows[2][9:].strip().split(",")
            ]
    except IndexError:
        pass
    try:
        if output_rows[3].startswith("medium: "):
            res["medium"] = [
                v.strip().split(" ")[0].strip()
                for v in output_rows[3][7:].strip().split(",")
            ]
    except IndexError:
        pass
    try:
        if output_rows[4].startswith("topic: "):
            res["topic"] = [
                v.strip().split(" ")[0].strip()
                for v in output_rows[4][6:].strip().split(",")
            ]
    except IndexError:
        pass
    try:
        if output_rows[5].startswith("addr: "):
            res["addr"] = [
                v.strip().split(" ")[0].strip()
                for v in output_rows[5][5:].strip().split(",")
            ]
    except IndexError:
        pass
    try:
        if output_rows[6].startswith("message: "):
            res["message"] = [
                v.strip().split(" ")[0].strip()
                for v in output_rows[6][8:].strip().split(",")
            ]
    except IndexError:
        pass
    try:
        if output_rows[7].startswith("source: "):
            res["source"] = [
                v.strip().split(" ")[0].strip()
                for v in output_rows[7][7:].strip().split(",")
            ]
    except IndexError:
        pass

    for key, value in res.items():
        if value == [""] or value == ["#UNK#"]:
            res[key] = ""
        while "#UNK#" in value:
            value.pop(value.index("#UNK#"))
        while type(value) == list and "" in value:
            value.pop(value.index(""))
        res[key] = value

    return res


# %%
def extract_roles():
    path = "./output/data/"

    for file in sorted(os.listdir(path)):
        if file.endswith(".zip"):
            continue
        file_content = {}

        with open(os.path.join(path, file), "r") as f:
            file_content = json.load(f)
            file_content["Outputs"]["Roles_text"] = {}

            for id, roles_for_sentence in file_content["Outputs"]["Roles"].items():
                file_content["Outputs"]["Roles_text"][id] = []

                if roles_for_sentence == []:
                    continue

                for roles_output in roles_for_sentence:
                    file_content["Outputs"]["Roles_text"][id].append([])

                    roles = extract_roles_from_output(roles_output[0])
                    file_content["Outputs"]["Roles_text"][id][-1].append(roles)

        with open(os.path.join(path, file), "w", encoding="utf8") as outfile:
            json.dump(file_content, outfile, indent=3)


# %%
extract_roles()

# %% [markdown]
# ## Mapping

# %%
def count_neighbors(i, seen, skip_index):
    res = 0
    if i - 2 >= 0 and i - 2 != skip_index:
        res += 1 if seen[i - 2] else 0
    if i - 1 >= 0 and i - 1 != skip_index:
        res += 1 if seen[i - 1] else 0
    if i + 1 < len(seen) and i + 1 != skip_index:
        res += 1 if seen[i + 1] else 0
    if i + 2 < len(seen) and i + 2 != skip_index:
        res += 1 if seen[i + 2] else 0
    return res

# %%
def calculate_neighborhood_swap(seen, tokens):
    for i, v in enumerate(seen):
        if not v:
            continue

        neigh_c_v = count_neighbors(i, seen, -1)
        neigh = [
            j
            for j, t in enumerate(tokens)
            if seen[j] == False and Levenshtein.distance(t, tokens[i]) <= 1
        ]
        neigh_c_other = [count_neighbors(n, seen, i) for n in neigh]
        if len(neigh_c_other) > 0:
            neigh_c_other_max = max(neigh_c_other)
            if neigh_c_other_max > neigh_c_v:
                return i, neigh[neigh_c_other.index(neigh_c_other_max)]

    return -1, -1


# %%
def map_output_list(output_list: list, ids: list, tokens: list, seen_old=None):
    res = []
    seen = [False] * len(tokens)
    if seen_old == None:
        seen_old = [False] * len(tokens)

    for output in output_list:
        indices = [
            i
            for i, v in enumerate(tokens)
            if v == output and seen[i] == False and seen_old[i] == False
        ]
        if len(indices) > 0:
            seen[indices[0]] = True
        if len(indices) == 0:
            indices = [
                i
                for i, v in enumerate(tokens)
                if seen[i] == False
                and seen_old[i] == False
                and Levenshtein.distance(output, v) <= 1
            ]
            if len(indices) > 0:
                seen[indices[0]] = True

    changed = True
    while changed:
        changed = False
        i, j = calculate_neighborhood_swap(seen, tokens)
        while i != j:
            seen[i] = False
            seen[j] = True
            changed = True
            i, j = calculate_neighborhood_swap(seen, tokens)

        for i in range(len(seen)):
            if (
                seen[i] == False
                and i != 0
                and i != len(seen) - 1
                and seen[i - 1]
                and seen[i + 1]
                and (
                    tokens[i] == ","
                    or tokens[i] == ":"
                    or tokens[i] == ";"
                    or tokens[i] == "-"
                )
            ):
                seen[i] = True
                changed = True

    for i in range(len(seen)):
        if seen[i]:
            res.append(str(ids[i]) + ":" + str(i))

    return res, [v or seen_old[i] for i, v in enumerate(seen)]


# %%
def map_outputs(ds):
    path = "./output/data/"

    for file in sorted(os.listdir(path)):
        if file.endswith(".zip"):
            continue
        file_content = {}

        with open(os.path.join(path, file), "r") as f:
            file_content = json.load(f)
            file_content["Annotations"] = []

            for cues_text, roles_text in zip(
                file_content["Outputs"]["Cues_text"].items(),
                file_content["Outputs"]["Roles_text"].items(),
            ):
                id, cues = cues_text
                id, roles_list = roles_text

                if cues == []:
                    continue

                tokens = ds.filter(
                    lambda r: r["FileName"] == file and r["SentenceId"] == int(id)
                )[0]["tokens_extended"]
                ids = ds.filter(
                    lambda r: r["FileName"] == file and r["SentenceId"] == int(id)
                )[0]["sentence_extended_ids"]

                seen_cues = None
                for cue, roles in zip(cues, roles_list):
                    roles = roles[0]

                    cue, seen_cues = map_output_list(cue, ids, tokens, seen_cues)

                    if cue != []:
                        addr, _ = map_output_list(
                            roles["addr"],
                            ids,
                            tokens,
                        )

                        evidence, _ = map_output_list(
                            roles["evidence"],
                            ids,
                            tokens,
                        )

                        medium, _ = map_output_list(
                            roles["medium"],
                            ids,
                            tokens,
                        )

                        message, _ = map_output_list(
                            roles["message"],
                            ids,
                            tokens,
                        )

                        source, _ = map_output_list(
                            roles["source"],
                            ids,
                            tokens,
                        )

                        topic, _ = map_output_list(
                            roles["topic"],
                            ids,
                            tokens,
                        )

                        ptc, _ = map_output_list(
                            roles["ptc"],
                            ids,
                            tokens,
                        )

                        annotation = {
                            "Addr": addr,
                            "Evidence": evidence,
                            "Medium": medium,
                            "Message": message,
                            "Source": source,
                            "Topic": topic,
                            "Cue": cue,
                            "PTC": ptc,
                        }
                        file_content["Annotations"].append(annotation)

        with open(os.path.join(path, file), "w", encoding="utf8") as outfile:
            json.dump(file_content, outfile, indent=3)

# %%
map_outputs(test_ds)


# %% [markdown]
# # Prepare zip file for submission

# %%
if os.path.exists("./output/data/submission.zip"):
    os.remove("./output/data/submission.zip")

temp_path = "./output/data/temp"
shutil.copytree("./output/data", temp_path)

for file in sorted(os.listdir(temp_path)):
    file_content = {}

    with open(os.path.join(temp_path, file), "r") as f:
        file_content = json.load(f)
        file_content.pop("Outputs")

    with open(os.path.join(temp_path, file), "w", encoding="utf8") as outfile:
        json.dump(file_content, outfile, indent=3)
shutil.make_archive(temp_path, "zip", temp_path)
shutil.move(
    temp_path + ".zip",
    "./output/data/submission.zip",
)
shutil.rmtree(temp_path)



