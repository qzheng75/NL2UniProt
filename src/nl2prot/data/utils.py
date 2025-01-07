from __future__ import annotations

import os
import warnings
from pathlib import Path

import torch
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from google.cloud import storage
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


def write_fasta(identifiers: list[str], sequences: list[str], file_path: str) -> None:
    """
    Write a list of identifiers and sequences to a fasta file

    Args:
        identifiers (list[str]): A list of identifiers.
        sequences (list[str]): A list of sequences.

    Raises:
        ValueError: If the lengths of the identifiers and sequences do not match.

    Example:
        >>> write_fasta(["identifier1", "identifier2"], ["sequence1", "sequence2"])
    """
    if len(identifiers) != len(sequences):
        raise ValueError("The number of identifiers and sequences must be the same.")

    records = [
        SeqRecord(Seq(sequence), id=name, description="")
        for name, sequence in zip(identifiers, sequences)
    ]
    SeqIO.write(records, file_path, "fasta")


def is_in_folder(file_path: str, folder_path: str) -> bool:
    """
    Check if a file is inside a folder or its subfolders.

    Args:
        file_path (str): Path to the file
        folder_path (str): Path to the folder to check

    Returns:
        bool: True if the file is in the folder or its subfolders, False otherwise
    """
    file = Path(file_path)
    folder = Path(folder_path)
    try:
        return folder in file.parents
    except Exception:
        return False


def download_from_gcs(bucket_name: str, src_folder: str, dest_folder: str) -> None:
    """
    Download all files from a GCS bucket folder to a local directory.

    Args:
        bucket_name: Name of the GCS bucket
        src_folder: Source folder path in the bucket
        dest_folder: Local destination folder path
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    all_blobs = bucket.list_blobs()

    for blob in all_blobs:
        if blob.name.endswith("/") or not is_in_folder(blob.name, src_folder):
            continue

        dest_path = os.path.join(dest_folder, blob.name)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        blob.download_to_filename(dest_path)


def upload_to_gcs(bucket_name: str, src_folder: str, dest_folder: str) -> None:
    """
    Upload all files from a local directory to a GCS bucket folder.

    Args:
        bucket_name: Name of the GCS bucket
        src_folder: Source folder path in the local directory
        dest_folder: Destination folder path in the bucket
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    for root, _, files in os.walk(src_folder):
        for file in files:
            local_path = os.path.join(root, file)
            blob_path = os.path.join(
                dest_folder, os.path.relpath(local_path, src_folder)
            )

            blob = bucket.blob(blob_path)
            blob.upload_from_filename(local_path)
