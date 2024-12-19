from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np
from dotenv import load_dotenv
from nl2prot.modules.evaluator import Evaluator
from nl2prot.validate.embedding import embed_descriptions
from scanpy import read

ID_TRANSFORMS = defaultdict(
    lambda: lambda x: x,
    {
        "|": lambda x: x.split("|")[0],
        "_": lambda x: x.split("_")[0],
        "-": lambda x: x.split("-")[0],
        ",": lambda x: x.split(",")[0],
    },
)


def parse_args():
    parser = argparse.ArgumentParser(description="Compute top-k")
    parser.add_argument(
        "--seq-embedding-file",
        type=str,
        required=True,
        help="Path to the sequence embedding file",
    )
    parser.add_argument(
        "--description-file",
        type=str,
        required=True,
        help="Path to the description file",
    )
    parser.add_argument(
        "--k", required=True, nargs="+", type=int, help="Top-k values to compute"
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
        "--id-transform", type=str, default="|", help="Delimiter to split the ID"
    )

    return parser.parse_args()


def main(args):
    assert args.description_file.endswith(
        ".json"
    ), "Description file must be a JSON file"

    with open(args.description_file, "r") as f:
        test_desc = json.load(f)

    names, desc = [], []
    for entry in test_desc:
        names.append(ID_TRANSFORMS[args.id_transform](entry["id"]))
        desc.append(entry["description"])

    accessions, embeddings = embed_descriptions(
        module_config_path=args.module_config_path,
        model_ckpt_path=args.model_ckpt_path,
        tokenizer_args=args.tokenizer_args,
        descriptions=desc,
        accessions=names,
    )
    assert (
        accessions is not None
    ), "Accessions should be provided in json. Report this issue."

    adata = read(args.seq_embedding_file)
    assert isinstance(adata.X, np.ndarray), "Sequence embedding must be a numpy array"
    ac2idx = {adata.obs["accession"].iloc[i]: i for i in range(len(adata))}

    evaluator = Evaluator(metric="TopKAcc", ks=args.k)
    results = evaluator.evaluate(
        desc_embedding=embeddings,
        seq_embedding=adata.X,
        ground_truth=np.array([ac2idx[s] for s in accessions]),
    )

    for key, value in results.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    load_dotenv()
    main(parse_args())
