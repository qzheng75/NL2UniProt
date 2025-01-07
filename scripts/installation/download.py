from __future__ import annotations

import argparse

from dotenv import load_dotenv
from nl2prot.data.utils import download_from_gcs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--download_data", type=bool, default=True, help="Download data from GCS"
    )
    parser.add_argument(
        "--data_dir", type=str, default="./", help="Directory to save data"
    )
    parser.add_argument(
        "--doanload_trained_model",
        type=bool,
        default=True,
        help="Download model from GCS",
    )
    parser.add_argument(
        "--trained_model_dir",
        type=str,
        default="./",
        help="Directory to save model",
    )
    return parser.parse_args()


def main():
    load_dotenv()
    args = parse_args()
    if args.download_data:
        download_from_gcs("dsgt-nl2uniprot", "raw_data", args.data_dir)
        print(f"Data downloaded successfully to {args.data_dir}")
    if args.doanload_trained_model:
        download_from_gcs("dsgt-nl2uniprot", "trained_models", args.trained_model_dir)
        print(f"Trained models downloaded successfully to {args.trained_model_dir}")


if __name__ == "__main__":
    main()
