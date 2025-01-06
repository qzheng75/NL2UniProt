from __future__ import annotations

import json
import random
from collections.abc import Callable
from itertools import chain
from typing import Any, override

from nl2prot.data.utils import load_fasta
from scanpy import read
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
    def __init__(self, raw_sequences: list[str], accessions: list[str]) -> None:
        self.idx2entry_map = {i: accessions[i] for i in range(len(accessions))}
        self.entry2idx_map = {v: k for k, v in self.idx2entry_map.items()}
        self.raw_sequences = raw_sequences

    def __len__(self) -> int:
        return len(self.raw_sequences)

    @override
    def __getitem__(self, idx: int) -> tuple[str, str]:
        return self.idx2entry_map[idx], self.raw_sequences[idx]


class DescDataset(Dataset):
    def __init__(
        self, descriptions: list[str], accessions: list[str] | None = None
    ) -> None:
        self.descriptions = descriptions
        self.accessions = accessions

    def __len__(self) -> int:
        return len(self.descriptions)

    @override
    def __getitem__(self, idx: int) -> tuple[str | None, str]:
        return self.accessions[idx] if self.accessions else None, self.descriptions[idx]


class RawDescSeqDataset(Dataset):
    def __init__(
        self, data_path: str, desc_path: str, use_copy: list[int] | None = None
    ):
        desc: dict[str, str] = {
            entry["id"]: entry["description"]
            for entry in json.load(open(desc_path, "r"))
        }
        if use_copy is not None:
            pattern_set = set(use_copy)
            desc = {
                key: value
                for key, value in desc.items()
                if int(key.split("|")[-1]) in pattern_set
            }

        desc_dict_with_names = merge_dictionaries(
            desc, desc, key_transform=lambda x: x.split("|")[0]
        )

        if data_path.endswith(".h5ad"):
            adata = read(data_path)
            name2seq_dict = {
                entry["accession"]: entry["seq"]
                for entry in adata.obs.to_dict("records")
            }
        elif data_path.endswith(".fasta"):
            entries = load_fasta(data_path)
            name2seq_dict = {entry[0]: entry[1] for entry in entries}
        else:
            raise ValueError("Invalid data file format")

        data = merge_dictionaries(
            name2seq_dict, desc_dict_with_names, allow_repeat_keys=True
        )
        data = dict(filter(lambda x: len(x[1]) != 1, data.items()))

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
    def __getitem__(self, idx: int) -> tuple[str, str, str]:
        return self.names[idx], self.sequences[idx], self.descriptions[idx]


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
