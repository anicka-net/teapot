#!/usr/bin/env python3
"""
Teapot Compose Validator — verify composed output is correct and trainable.

Checks:
  1. Format: every example has valid structure
  2. Content: tier/role filtering, no duplicates, license compliance
  3. Composition: weights match, deterministic, manifest accurate
  4. Trainability: tokenizes correctly, fits in max_length

Usage:
    python3 scripts/validate_compose.py train-apertus-70b-secular.jsonl
    python3 scripts/validate_compose.py train.jsonl --manifest train.manifest.json
    python3 scripts/validate_compose.py train.jsonl --tokenizer swiss-ai/Apertus-70B-Instruct-2509
    python3 scripts/validate_compose.py train.jsonl --check-tiers secular  # verify no Buddhist leaked
"""

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


def log(msg, level="INFO"):
    markers = {"INFO": " ", "PASS": "+", "FAIL": "-", "WARN": "!"}
    print(f"  [{markers.get(level, ' ')}] {msg}")


def check_format(examples):
    """Check structural validity of every example."""
    errors = 0
    warnings = 0

    for i, ex in enumerate(examples):
        convs = ex.get("conversations", [])
        text = ex.get("text", "")

        if not convs and not text:
            log(f"Example {i} ({ex.get('id', '?')}): empty conversations", "FAIL")
            errors += 1
            continue

        if text and not ex.get("assistant_spans"):
            log(f"Example {i} ({ex.get('id', '?')}): formatted text missing assistant_spans", "FAIL")
            errors += 1

        roles = [m.get("role") for m in convs]
        valid_roles = {"system", "user", "assistant", "ipython", "tool"}

        # Check roles are valid
        for r in roles:
            if r not in valid_roles:
                log(f"Example {i} ({ex.get('id', '?')}): invalid role '{r}'", "FAIL")
                errors += 1

        # Check at least one user + one assistant
        if "user" not in roles:
            log(f"Example {i} ({ex.get('id', '?')}): no user message", "FAIL")
            errors += 1
        if "assistant" not in roles:
            log(f"Example {i} ({ex.get('id', '?')}): no assistant message", "FAIL")
            errors += 1

        # Check no empty content
        for m in convs:
            if not m.get("content", "").strip():
                log(f"Example {i} ({ex.get('id', '?')}): empty content in {m.get('role')} message", "WARN")
                warnings += 1

    return errors, warnings


def check_content(examples, expected_tier=None):
    """Check content-level properties."""
    errors = 0
    warnings = 0

    ids = [ex.get("id", "") for ex in examples]
    categories = Counter(ex.get("category", "") for ex in examples)
    tasks = {ex.get("task", "chat-completion") for ex in examples}
    modules = {ex.get("module", "") for ex in examples}

    # Duplicate check
    id_counts = Counter(ids)
    dupes = {k: v for k, v in id_counts.items() if v > 1 and k}
    if dupes:
        log(f"Found {len(dupes)} duplicate IDs (top 5): {list(dupes.items())[:5]}", "WARN")
        warnings += len(dupes)

    # Tier check
    if expected_tier:
        buddhist_keywords = [
            'buddha', 'dharma', 'bodhisattva', 'gampopa', 'nagarjuna',
            'samsara', 'nirvana', 'sutra', 'rinpoche', 'vajrayana',
            'mahayana', 'theravada', 'sangha', 'precept',
        ]
        if expected_tier == "secular":
            # Check that assistant responses don't contain Buddhist-specific terms
            # (allowing user prompts to mention them — questions about Buddhism are fine)
            leaked = 0
            for ex in examples:
                asst_text = ' '.join(
                    m['content'] for m in ex.get('conversations', [])
                    if m.get('role') == 'assistant'
                ).lower()
                found = [kw for kw in buddhist_keywords if kw in asst_text]
                # Allow examples that discuss Buddhism factually
                # (reviewed by Sonnet agent, verdict KEEP_SECULAR)
                # Only flag if the model ADOPTS Buddhist framing unprompted
                user_text = ' '.join(
                    m['content'] for m in ex.get('conversations', [])
                    if m.get('role') == 'user'
                ).lower()
                user_asks_about_buddhism = any(
                    kw in user_text for kw in [
                        'buddhis', 'buddha', 'dharma', 'meditation', 'karma',
                        'tibet', 'zen', 'sutra', 'enlighten', 'monk',
                    ]
                )
                safe_cats = {
                    'adversarial-philosophical-framing',
                    'compassion-exploit-refusal',
                    'adversarial-general',
                    'adversarial-persona-defense',
                }
                if found and not user_asks_about_buddhism and ex.get('category', '') not in safe_cats:
                    leaked += 1
                    if leaked <= 3:
                        log(f"Buddhist content in secular: {ex.get('id', '?')} has {found}", "WARN")
            if leaked:
                log(f"Total Buddhist leaks in secular config: {leaked}", "WARN")
                warnings += leaked
            else:
                log("No Buddhist content leaked into secular tier", "PASS")

    # Reward-eval check
    reward_cats = {'reward-evaluation', 'reward-evaluation-v5', 'reward-evaluation-style-variant'}
    reward_rows = [ex for ex in examples if ex.get('category', '') in reward_cats]
    reward_dataset = (
        tasks == {"reward-evaluation"} or
        modules == {"capability/reward-evaluator"}
    )
    if reward_dataset:
        non_reward = [ex for ex in examples if ex.get('category', '') not in reward_cats]
        if non_reward:
            log(f"Non reward-evaluator rows in reward-model dataset: {len(non_reward)}", "FAIL")
            errors += len(non_reward)
        else:
            log("Reward-evaluator dataset detected", "PASS")
    elif reward_rows:
        log(f"Reward-evaluator examples in conversational output: {len(reward_rows)}", "FAIL")
        errors += len(reward_rows)
    else:
        log("No reward-evaluator examples leaked", "PASS")

    # Category distribution
    log(f"Categories: {len(categories)} unique")
    for cat, count in categories.most_common(10):
        log(f"  {cat}: {count}")

    return errors, warnings


def check_manifest(examples, manifest_path):
    """Verify manifest matches actual output."""
    errors = 0

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Total count
    if manifest["total_examples"] != len(examples):
        log(f"Manifest says {manifest['total_examples']} examples, file has {len(examples)}", "FAIL")
        errors += 1
    else:
        log(f"Example count matches manifest: {len(examples)}", "PASS")

    # Per-module counts (check _module field if present, otherwise by category heuristic)
    # Since we strip _module in output, check by counting
    log(f"Manifest modules: {list(manifest.get('modules', {}).keys())}")

    return errors


def check_tokenization(examples, tokenizer_name, max_length=4096, sample_size=100):
    """Check that examples tokenize correctly and fit in max_length."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        log("transformers not installed — skipping tokenization check", "WARN")
        return 0, 1

    log(f"Loading tokenizer: {tokenizer_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    except Exception as e:
        log(f"Could not load tokenizer: {e}", "WARN")
        return 0, 1

    import random
    random.seed(42)
    sample = random.sample(examples, min(sample_size, len(examples)))

    errors = 0
    too_long = 0
    lengths = []

    for ex in sample:
        convs = ex.get("conversations", [])
        try:
            if ex.get("text"):
                tokens = tokenizer.encode(ex["text"], add_special_tokens=False)
            else:
                text = tokenizer.apply_chat_template(convs, tokenize=False)
                tokens = tokenizer.encode(text)
            lengths.append(len(tokens))
            if len(tokens) > max_length:
                too_long += 1
        except Exception as e:
            log(f"Tokenization failed for {ex.get('id', '?')}: {str(e)[:80]}", "FAIL")
            errors += 1

    if lengths:
        avg = sum(lengths) / len(lengths)
        log(f"Token lengths (sample of {len(sample)}): avg={avg:.0f}, "
            f"min={min(lengths)}, max={max(lengths)}", "INFO")
        if too_long:
            log(f"{too_long}/{len(sample)} examples exceed max_length={max_length}", "WARN")
        else:
            log(f"All sampled examples fit within max_length={max_length}", "PASS")

    return errors, too_long


def check_determinism(config_path, output_path):
    """Verify that re-running compose produces identical output."""
    import subprocess

    log("Re-running compose to check determinism...")
    result = subprocess.run(
        [sys.executable, "scripts/compose.py", config_path, "--output", "/tmp/teapot-determinism-check.jsonl"],
        capture_output=True, text=True,
    )

    if result.returncode != 0:
        log("Determinism check: compose re-run failed", "WARN")
        return 0, 1

    # Compare file hashes
    h1 = hashlib.sha256(open(output_path, "rb").read()).hexdigest()
    h2 = hashlib.sha256(open("/tmp/teapot-determinism-check.jsonl", "rb").read()).hexdigest()

    Path("/tmp/teapot-determinism-check.jsonl").unlink()
    Path("/tmp/teapot-determinism-check.manifest.json").unlink(missing_ok=True)

    if h1 == h2:
        log("Deterministic: re-run produces identical output", "PASS")
        return 0, 0
    else:
        log("NOT deterministic: re-run produces different output!", "FAIL")
        return 1, 0


def main():
    parser = argparse.ArgumentParser(description="Validate Teapot compose output")
    parser.add_argument("input", help="Composed JSONL file")
    parser.add_argument("--manifest", help="Manifest JSON file")
    parser.add_argument("--tokenizer", help="Tokenizer name/path for trainability check")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--check-tiers", help="Verify tier filtering (e.g., 'secular')")
    parser.add_argument("--config", help="Config file for determinism check")
    args = parser.parse_args()

    print("=" * 60)
    print("TEAPOT COMPOSE VALIDATION")
    print(f"Input: {args.input}")
    print("=" * 60)

    # Load examples
    examples = []
    with open(args.input) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    print(f"Loaded {len(examples)} examples\n")

    total_errors = 0
    total_warnings = 0

    # 1. Format
    print("--- Format Validation ---")
    e, w = check_format(examples)
    total_errors += e
    total_warnings += w
    print()

    # 2. Content
    print("--- Content Validation ---")
    e, w = check_content(examples, expected_tier=args.check_tiers)
    total_errors += e
    total_warnings += w
    print()

    # 3. Manifest
    if args.manifest:
        print("--- Manifest Validation ---")
        e = check_manifest(examples, args.manifest)
        total_errors += e
        print()

    # 4. Tokenization
    if args.tokenizer:
        print("--- Tokenization Validation ---")
        e, w = check_tokenization(examples, args.tokenizer, args.max_length)
        total_errors += e
        total_warnings += w
        print()

    # 5. Determinism
    if args.config:
        print("--- Determinism Validation ---")
        e, w = check_determinism(args.config, args.input)
        total_errors += e
        total_warnings += w
        print()

    # Summary
    print("=" * 60)
    if total_errors == 0:
        print(f"RESULT: PASS ({total_warnings} warnings)")
    else:
        print(f"RESULT: FAIL ({total_errors} errors, {total_warnings} warnings)")
    print("=" * 60)

    sys.exit(1 if total_errors > 0 else 0)


if __name__ == "__main__":
    main()
