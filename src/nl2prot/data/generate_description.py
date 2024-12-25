from __future__ import annotations

import json
import logging
import os

import torch
import transformers

DEFAULT_PROMPT = "Return empty string if there isn't enough information.\
 Use professtional terminology\
 to summarize this protein sequence in {length:d} sentence. Start with:\
 'I want a protein with ...'. IMPORTANT: Don't give the exact name or\
 numbers or mention not available features."


def load_model(model_name=None):
    if model_name is None:
        model_name = "meta-llama/Llama-3.1-8B-Instruct"

    cache_dir = os.environ["MODEL_CACHE"]
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, cache_dir=cache_dir
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cuda", cache_dir=cache_dir
    )

    torch.set_float32_matmul_precision("high")
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, model


def generate_description(
    tokenizer,
    model,
    ids: list[str],
    prot_info: list[str],
    n_sentences=2,
    prompt=DEFAULT_PROMPT,
    batch_size=4,
    verbose=1000,
    max_new_tokens=256,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    responses = []
    messages = [
        [
            {"role": "system", "content": prompt.format(length=n_sentences)},
            {"role": "user", "content": info},
        ]
        for info in prot_info
    ]
    verbose_threshold = verbose
    logging.info(f"Generating descriptions for {len(ids)} proteins")

    for i in range(0, len(ids), batch_size):
        if verbose > 0 and i >= verbose_threshold:
            logging.info(f"Processing {i} out of {len(ids)}")
            verbose_threshold += verbose
        batch_ids = ids[i : i + batch_size]
        batch_messages = messages[i : i + batch_size]

        prompts = [
            tokenizer.apply_chat_template(msg, tokenize=False) for msg in batch_messages
        ]

        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, do_sample=True, max_new_tokens=max_new_tokens
            )

        batch_responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for j, response in enumerate(batch_responses):
            generated_text = response.split("\n")[-1].strip()
            responses.append({"id": batch_ids[j], "description": generated_text})

    return responses


def save_description(
    output: list[dict],
    output_path: str,
):
    with open(output_path, "w") as f:
        json.dump(output, f, indent=4)
