from __future__ import annotations

import json
import random
from collections import namedtuple
from collections.abc import Callable
from itertools import chain
from typing import Any, override

import torch
from scanpy import read
from torch import Tensor
from torch.utils.data import Dataset


def merge_dictionaries(
    dict1: dict[str, Any],
    dict2: dict[str, Any],
    key_transform: Callable[[str], str] = lambda x: x,
    allow_repeat_keys: bool = False,
) -> dict[str, Any]:
    result = {}
    visited_ids = set()
    for key, value in chain(dict1.items(), dict2.items()):
        if not allow_repeat_keys and key in visited_ids:
            continue
        visited_ids.add(key)
        prefix = key_transform(key)
        if prefix not in result:
            result[prefix] = [value]
        else:
            result[prefix].append(value)

    return result


def apply_key_transform(
    orig_dict: dict[str, Any], key_map: dict[str, Any]
) -> dict[str, Any]:
    updated_result = {}
    for key, value in orig_dict.items():
        if key in key_map:
            new_key = key_map[key]
            updated_result[new_key] = value
        else:
            updated_result[key] = value
    return updated_result


class ProtDataset(Dataset):
    def __init__(self, x, names) -> None:
        self.idx2entry_map = {i: names[i] for i in range(len(x))}
        self.entry2idx_map = {v: k for k, v in self.idx2entry_map.items()}
        self.seq_embed = torch.tensor(x, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.seq_embed)

    @override
    def __getitem__(self, idx: int) -> tuple[Tensor, str]:
        return self.seq_embed[idx], self.idx2entry_map[idx]


class EmbeddedDescDataset(Dataset):
    def __init__(
        self,
        adata_path: str,
        desc_embed_path: str,
    ) -> None:
        adata = read(adata_path)
        desc_embed = torch.load(desc_embed_path)

        desc_dict = merge_dictionaries(
            desc_embed, desc_embed, key_transform=lambda x: x.split("|")[0]
        )
        self.desc_name_tuples = []
        EmbEntry = namedtuple("EmbEntry", ["name", "emb"])
        for key, item in desc_dict.items():
            for emb in item:
                self.desc_name_tuples.append(EmbEntry(key, emb))

        self.prot_dataset = ProtDataset(adata.X, adata.obs["accession"])

    def __len__(self) -> int:
        return len(self.desc_name_tuples)

    @override
    def __getitem__(self, idx: int) -> dict[str, Any]:
        name, emb = self.desc_name_tuples[idx]
        prot_emb, prot_name = self.prot_dataset[self.prot_dataset.entry2idx_map[name]]
        return {
            "prot_emb": prot_emb,
            "desc_emb": emb,
            "prot_name": prot_name,
            "desc_name": name,
        }


class RawDescSeqDataset(Dataset):
    def __init__(self, adata_path, desc_path):
        desc = {
            entry["id"]: entry["description"]
            for entry in json.load(open(desc_path, "r"))
        }
        desc_dict_with_names = merge_dictionaries(
            desc, desc, key_transform=lambda x: x.split("|")[0]
        )

        self.adata = read(adata_path)
        name2seq_dict = {
            entry["accession"]: entry["seq"]
            for entry in self.adata.obs.to_dict("records")
        }

        data = merge_dictionaries(
            name2seq_dict, desc_dict_with_names, allow_repeat_keys=True
        )
        data = dict(filter(lambda x: len(x[1]) != 1, data.items()))

        # self.names = list(data.keys())
        # vals = list(data.values())
        # self.sequences = [entry[0] for entry in vals]
        # self.descriptions = [entry[1] for entry in vals]

        names, sequences, descriptions = [], [], []
        for key, val in data.items():
            if len(val) == 1:
                continue
            for desc in val[1]:
                names.append(key)
                sequences.append(val[0])
                descriptions.append(desc)

        self.names = names
        self.sequences = sequences
        self.descriptions = descriptions
        zipped = list(zip(self.names, self.sequences, self.descriptions))
        random.shuffle(zipped)
        self.names, self.sequences, self.descriptions = zip(*zipped)

    def __len__(self):
        return len(self.names)

    @override
    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {
            "name": self.names[idx],
            "sequence": self.sequences[idx],
            "description": self.descriptions[idx],
        }


# DEEPLOC_CLASSES = [
#     'Membrane', 'Cytoplasm', 'Nucleus', 'Extracellular',
#     'Cell membrane', 'Mitochondrion', 'Plastid', 'Endoplasmic reticulum',
#     'Lysosome/Vacuole', 'Golgi apparatus', 'Peroxisome'
# ]

# DEEPLOC_CLASSES = [
#     'Membrane'
# ]

# class DeepLocDataset(Dataset):
#     def __init__(
#         self,
#         deeploc_csv_path: str,
#         use_folds: list[str] = None
#     ):
#         self.data = pd.read_csv(deeploc_csv_path)
#         if use_folds is not None:
#             self.data = self.data[self.data['Partition'].isin(use_folds)]

#         self.data = self.data.reset_index(drop=True)

#         self.labels = []
#         for class_name in DEEPLOC_CLASSES:
#             if class_name in self.data.columns:
#                 self.labels.append(self.data[class_name].values)
#             # else:
#             #     self.labels.append(np.zeros(len(self.data)))

#         if len(self.labels) > 1:
#             self.labels = torch.tensor(np.stack(self.labels, axis=1))
#         else:
#             self.labels = torch.tensor(self.labels).view(-1, 1)

#         if 'fasta' in self.data.columns: # Test dataset
#             self.sequences = self.data['fasta'].values
#         elif 'Sequence' in self.data.columns: # Train/Val dataset
#             self.sequences = self.data['Sequence'].values
#         else:
#             raise ValueError("No sequence column found in the dataset")

#         if 'ACC' in self.data.columns:
#             self.names = self.data['ACC'].values
#         elif 'sid' in self.data.columns:
#             self.names = self.data['sid'].values
#         else:
#             raise ValueError("No name column found in the dataset")

#         self.data_items = [
#             {
#                 'name': name,
#                 'sequence': seq,
#                 'labels': label
#             } for name, seq, label in zip(self.names, self.sequences, self.labels)
#         ]

#     def __len__(self) -> int:
#         return len(self.data_items)

#     def __getitem__(self, idx: int) -> dict[str, Tensor]:
#         return self.data_items[idx]
