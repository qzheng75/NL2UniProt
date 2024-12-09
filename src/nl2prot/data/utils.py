from __future__ import annotations

import json
import os
import warnings
from collections.abc import Callable
from typing import Literal

import torch
from Bio import SeqIO
from torch.utils.data import Dataset, random_split


def split_dataset(
    dataset: Dataset,
    train_size: float = 0.8,
    valid_size: float = 0.1,
    test_size: float = 0.1,
    seed: int = 42,
) -> tuple[Dataset, Dataset, Dataset]:
    if train_size + valid_size + test_size > 1:
        warnings.warn(
            "Invalid sizes detected (ratios add up to larger than one)."
            + " Using default split of 0.8/0.05/0.15."
        )
        train_size, valid_size, test_size = 0.8, 0.05, 0.15

    dataset_size = len(dataset)  # type: ignore

    train_len = int(train_size * dataset_size)
    valid_len = int(valid_size * dataset_size)
    test_len = int(test_size * dataset_size)
    unused_len = dataset_size - train_len - valid_len - test_len

    torch.manual_seed(seed)
    (train_dataset, val_dataset, test_dataset, _) = random_split(
        dataset,
        [train_len, valid_len, test_len, unused_len],
    )

    return train_dataset, val_dataset, test_dataset


def load_embedding_str(
    embedding_path: str,
    sequence_identifiers: list[str],
    return_format: Literal["original", "id_repr_dict"],
    not_found_policy: Literal["raise", "ignore"] = "raise",
) -> list[dict] | dict:
    """
    Load sequence embeddings from a file and return them in the specified format.

    Args:
        embedding_path (str): The path to the file containing the sequence embeddings.
        sequence_identifiers (list[str]):
            A list of sequence identifiers to retrieve embeddings for.
        return_format (Literal['original', 'id_repr_dict']):
            The desired format of the returned embeddings.
            'original': Return a list of dictionaries,
            where each dictionary has a key=identifier and value=embedding.
            'id_repr_dict': Return a dictionary with key=identifier and value=embedding.

    Returns:
        list[dict] | dict: The loaded embeddings in the specified format.

    Raises:
        ValueError: If any sequence identifier is not found in the saved embeddings.
    """
    with open(embedding_path, "r") as f:
        embeddings = json.load(f)

    matches = [] if return_format == "original" else {}

    for seq in sequence_identifiers:
        match = next((d for d in embeddings if d["entry_id"] == seq), None)
        if match is None:
            if not_found_policy == "ignore":
                continue
            raise ValueError(f"Sequence {seq} not found in the saved embeddings")
        if return_format == "original":
            assert isinstance(matches, list)
            matches.append(match)
        else:
            assert isinstance(matches, dict)
            matches[seq] = match["mean_representations"]

    return matches


def load_embedding_fasta(
    embedding_path: str,
    fasta_path: str,
    identifier: Literal["label", "sequence"],
    label_parse_fn: Callable[[str], str] = lambda x: x,
    return_format: Literal["original", "id_repr_dict"] = "id_repr_dict",
    not_found_policy: Literal["raise", "ignore"] = "raise",
) -> list[dict] | dict:
    """
    Load sequence embeddings from a FASTA file.

    Args:
        embedding_path (str):
            The path to the file containing the sequence embeddings.
        fasta_path (str):
            The path to the FASTA file containing the sequence identifiers.
        identifier (Literal['label', 'sequence']): The type of identifier to use.
            'label' uses the label_parse_fn to parse the sequence identifier.
            'sequence' uses the sequence itself as the identifier.
        label_parse_fn (Callable[[str], str], optional):
            A function to parse the sequence identifier.
            This function is only used if the identifier is set to 'label'.
            Defaults to `lambda x: x`.
        return_format (Literal['original', 'id_repr_dict'], optional):
            The format of the returned embeddings.
            'original' returns a list of dictionaries,
                where each dictionary has the identifier as the key
                and the embedding as the value.
            'id_repr_dict' returns a dictionary with the identifier as the key
                and the embedding as the value. Defaults to 'id_repr_dict'.

    Returns:
        list[dict] or dict: The loaded sequence embeddings.
            If 'return_format' is set to 'original', a list of dictionaries is returned.
            If 'return_format' is set to 'id_repr_dict', a dictionary is returned.

    Raises:
        ValueError: If the 'identifier' argument is not set to 'label' or 'sequence'.
    """
    fasta_entries = load_fasta(fasta_path)

    if identifier == "label":
        sequence_identifiers = [label_parse_fn(id) for id, _ in fasta_entries]
    elif identifier == "sequence":
        sequence_identifiers = [seq for _, seq in fasta_entries]
    else:
        raise ValueError(f"Invalid value for 'identifier': {identifier}")

    return load_embedding_str(
        embedding_path, sequence_identifiers, return_format, not_found_policy
    )


def save_embeddings(
    embeddings: list[dict],
    output_path: str,
) -> None:
    """
    Save the embeddings to a JSON file.

    Args:
        embeddings (list[dict]):
            A list of dictionaries representing the embeddings.
        output_path (str):
            The path to the directory where the output file will be saved.

    Returns:
        None
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(embeddings, f)


def load_fasta(file_path: str) -> list[tuple[str, str]]:
    """
    Load a fasta file and return a list of tuples,
        each tuple containing the identifier and sequence

    Args:
        file_path (str): The path to the fasta file.

    Returns:
        list[tuple[str, str]]: A list of tuples,
            where each tuple contains the identifier and sequence.

    Raises:
        FileNotFoundError: If the specified file_path does not exist.

    Example:
        >>> load_fasta("path/to/fasta_file.fasta")
        [('identifier1', 'sequence1'), ('identifier2', 'sequence2'), ...]
    """
    entries = []
    with open(file_path, "r") as file:
        entries.extend(
            (record.id, str(record.seq)) for record in SeqIO.parse(file, "fasta")
        )
    return entries
