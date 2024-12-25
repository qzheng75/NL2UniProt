from __future__ import annotations

import argparse
import json
import os

from dotenv import load_dotenv
from nl2prot.validate.embedding import embed_sequences


def parse_args():
    parser = argparse.ArgumentParser(
        description="Embed sequences using a trained model"
    )
    parser.add_argument(
        "--module-config-path",
        type=str,
        required=True,
        help="Path to the module config file",
    )
    parser.add_argument(
        "--model-ckpt-dir",
        type=str,
        required=True,
        help="Path to the model checkpoint file",
    )
    parser.add_argument(
        "--tokenizer-args",
        type=json.loads,
        help="Arguments for the tokenizer",
        default='{"pretrained_model_name_or_path": "facebook/esm2_t12_35M_UR50D"}',
    )
    parser.add_argument(
        "--fasta-file", type=str, help="Path to a fasta file with sequences to embed"
    )
    parser.add_argument(
        "--raw-sequences", nargs="+", type=str, help="List of raw sequences to embed"
    )
    parser.add_argument(
        "--accessions", nargs="+", type=str, help="List of accessions for raw sequences"
    )
    return parser.parse_args()


def main(args):
    model_ckpt_dir = args.model_ckpt_dir
    model_ckpt_path = os.path.join(model_ckpt_dir, "best_state.pt")
    output_file = os.path.join(model_ckpt_dir, "seq_embed.h5ad")

    if not os.path.exists(model_ckpt_path):
        raise FileNotFoundError(f"Model checkpoint file not found: {model_ckpt_path}")

    _, embeddings = embed_sequences(
        model_ckpt_path,
        args.model_ckpt_path,
        args.tokenizer_args,
        args.fasta_file,
        args.raw_sequences,
        args.accessions,
        output_file,
    )
    print(f"Finished embedding {len(embeddings)} sequences")


if __name__ == "__main__":
    load_dotenv()
    args = parse_args()
    main(args)
