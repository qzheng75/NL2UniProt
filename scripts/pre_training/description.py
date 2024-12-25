from __future__ import annotations

import argparse
import logging

import pandas as pd
from dotenv import load_dotenv
from nl2prot.data import generate_description as gd

logging.basicConfig(level=logging.INFO)
logging.getLogger("transformers").setLevel(logging.ERROR)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_desc", type=int, default=1)
    parser.add_argument("--sample", type=int, default=-1)
    parser.add_argument(
        "--output_path", type=str, default="raw_data/misc/descriptions.json"
    )
    parser.add_argument("--verbose", type=int, default=-1)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    return parser.parse_args()


if __name__ == "__main__":
    load_dotenv()

    args = get_args()
    df = pd.read_csv("data/uniprot_processed.tsv", sep="\t")
    df = df.drop(columns=["Entry Name", "Gene Names", "Sequence", "Length"], axis=1)

    all_ids = []
    all_prot_info = []
    n_desc = args.n_desc

    if args.sample > 0:
        assert args.sample <= len(df)
        df = df.sample(args.sample)

    ids = df["Entry"].tolist()
    df = df.drop(columns=["Entry", "Mass"])

    for i in range(n_desc):
        all_ids.extend(map(lambda x: x + f"|{i}", ids))
        all_prot_info.extend([df.iloc[j].to_json() for j in range(len(df))])

    pmt = (
        "Return empty string if there isn't enough information. "
        + "Don't give the exact name or numbers. Use layman language to "
        + "summarize this protein sequence in 1 sentence. Start with: "
        + "'I want a protein with ...'. Don't mention not available features."
    )
    tokenizer, model = gd.load_model()
    output = gd.generate_description(
        tokenizer,
        model,
        all_ids,
        all_prot_info,
        prompt=pmt,
        verbose=args.verbose,
        max_new_tokens=args.max_new_tokens,
    )
    gd.save_description(output, args.output_path)
