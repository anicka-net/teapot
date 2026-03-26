#!/usr/bin/env python3
"""Shared training utilities for Teapot backends."""

import json

import torch
from torch.utils.data import Dataset

from teapot.templates import TEMPLATES, format_conversation


class FormattedDataset(Dataset):
    """Dataset that reads compose output and preserves assistant-only masking."""

    def __init__(self, data_path, tokenizer, max_length=4096, template=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        with open(data_path) as f:
            for line in f:
                if not line.strip():
                    continue
                ex = json.loads(line)

                if "text" in ex and ex["text"]:
                    text = ex["text"]
                    spans = ex.get("assistant_spans", [])
                else:
                    convs = ex.get("conversations", ex.get("messages", []))
                    if template:
                        text, spans = format_conversation(convs, template, thinking=True)
                        if text is None:
                            continue
                    else:
                        text = "\n".join(m.get("content", "") for m in convs)
                        spans = []

                self.examples.append({"text": text, "spans": spans})

        print(f"Loaded {len(self.examples)} examples from {data_path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        text = ex["text"]
        spans = ex["spans"]

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_offsets_mapping=True,
        )

        input_ids = encoding["input_ids"]
        offsets = encoding.get("offset_mapping", [])
        labels = [-100] * len(input_ids)

        if spans and offsets:
            for token_idx, (tok_start, tok_end) in enumerate(offsets):
                if tok_start == tok_end:
                    continue
                for span_start, span_end in spans:
                    if tok_start >= span_start and tok_end <= span_end:
                        labels[token_idx] = input_ids[token_idx]
                        break
        else:
            labels = list(input_ids)

        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(encoding["attention_mask"]),
            "labels": torch.tensor(labels),
        }


def collate_fn(batch):
    """Pad batch to same length."""
    max_len = max(len(ex["input_ids"]) for ex in batch)

    input_ids = []
    attention_mask = []
    labels = []

    for ex in batch:
        pad_len = max_len - len(ex["input_ids"])
        input_ids.append(torch.cat([ex["input_ids"], torch.zeros(pad_len, dtype=torch.long)]))
        attention_mask.append(torch.cat([ex["attention_mask"], torch.zeros(pad_len, dtype=torch.long)]))
        labels.append(torch.cat([ex["labels"], torch.full((pad_len,), -100, dtype=torch.long)]))

    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": torch.stack(labels),
    }


def verify_template_tokens(tokenizer, template):
    """Verify Teapot-owned template control tokens are registered as single special tokens."""
    if not template or template == "auto":
        return True

    template_meta = TEMPLATES.get(template)
    if not template_meta:
        return True

    critical = template_meta.get("special_tokens", [])
    if not critical:
        return True

    all_special_tokens = set(getattr(tokenizer, "all_special_tokens", []))
    ok = True

    for token in critical:
        ids = tokenizer.encode(token, add_special_tokens=False)
        is_special = token in all_special_tokens
        if len(ids) != 1 or not is_special:
            print(
                f"ERROR: {token} encodes as {ids} and "
                f"{'is not' if not is_special else 'is'} registered as a special token"
            )
            ok = False
        else:
            print(f"  OK: {token} → {ids}")

    if not ok:
        print("\nERROR: Template control tokens are not configured safely.")
        print("Refusing to start training with a broken chat format.")
        print("Check tokenizer_config.json / added_tokens.json for this model.")

    return ok
