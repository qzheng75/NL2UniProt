from __future__ import annotations

import argparse
import json

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
        "--model-ckpt-path",
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
    parser.add_argument("--output-file", type=str, help="Path to the output file")
    return parser.parse_args()


def main(args):
    _, embeddings = embed_sequences(
        args.module_config_path,
        args.model_ckpt_path,
        args.tokenizer_args,
        args.fasta_file,
        args.raw_sequences,
        args.accessions,
        args.output_file,
    )
    print(f"Finished embedding {len(embeddings)} sequences")


if __name__ == "__main__":
    load_dotenv()
    args = parse_args()
    main(args)
