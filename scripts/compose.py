#!/usr/bin/env python3
"""
Teapot Compose — merge module data into a training-ready JSONL.

Reads a config file, runs each module's prepare script, merges the
outputs with configured weights, applies license filtering, and
produces a shuffled train.jsonl + manifest.json.

Usage:
    python3 scripts/compose.py configs/karma-electric.config
    python3 scripts/compose.py configs/defconfig --output train-custom.jsonl
    python3 scripts/compose.py configs/karma-electric.config --dry-run
"""

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

TEAPOT_ROOT = Path(__file__).resolve().parents[1]


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_config(config_path):
    """Parse a Teapot config file (YAML format)."""
    import yaml

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    config = {
        "modules": {},
        "base_model": None,
        "licenses_allowed": [],
        "output": "train.jsonl",
        "seed": 42,
        "chat_template": "auto",
    }

    if not raw:
        return config

    # Base model
    base = raw.get("base", {})
    config["base_model"] = base.get("model")

    # Modules — can be dict of name: true/false
    modules_raw = raw.get("modules", {})
    weights_raw = raw.get("training", {}).get("weights", {})

    for module_name, enabled in modules_raw.items():
        config["modules"][module_name] = {
            "enabled": bool(enabled),
            "weight": weights_raw.get(module_name, 1.0),
        }

    # Licenses
    license_cfg = raw.get("license", {})
    allowed = license_cfg.get("allowed", [])
    if allowed == "all" or allowed == ["all"]:
        config["licenses_allowed"] = None  # None = no filtering
    elif isinstance(allowed, str):
        config["licenses_allowed"] = [allowed]
    else:
        config["licenses_allowed"] = allowed

    # Chat template
    training = raw.get("training", {})
    config["chat_template"] = training.get("chat_template", "auto")

    # Output
    output = raw.get("output", {})
    if isinstance(output, dict):
        config["output"] = output.get("file", "train.jsonl")
    elif isinstance(output, str):
        config["output"] = output

    # Seed
    config["seed"] = training.get("seed", 42)

    return config


def find_module(module_name):
    """Find a module's directory and YAML config."""
    # Convert safety/consequence → modules/safety/consequence/
    parts = module_name.split("/")
    module_dir = TEAPOT_ROOT / "modules" / "/".join(parts)

    if not module_dir.exists():
        log(f"  WARNING: Module directory not found: {module_dir}")
        return None, None

    yaml_path = module_dir / "module.yaml"
    if not yaml_path.exists():
        log(f"  WARNING: module.yaml not found: {yaml_path}")
        return module_dir, None

    # Parse YAML (simple — we don't need full YAML parser for this)
    # For now, just return the path and let prepare.py handle it
    return module_dir, yaml_path


def run_prepare(module_dir, module_name, chat_template="chatml"):
    """Run a module's prepare.py script."""
    prepare_script = module_dir / "prepare.py"

    if not prepare_script.exists():
        log(f"  WARNING: No prepare.py for {module_name}")
        # Check if data already exists
        data_dir = module_dir / "data"
        if data_dir.exists():
            jsonl_files = list(data_dir.glob("*.jsonl"))
            if jsonl_files:
                log(f"  Using existing data: {jsonl_files[0].name}")
                return jsonl_files[0]
        return None

    log(f"  Running prepare.py for {module_name}...")
    args = [sys.executable, str(prepare_script)]

    # Read prepare args from module.yaml if present
    yaml_path = module_dir / "module.yaml"
    if yaml_path.exists():
        import yaml
        mod_cfg = yaml.safe_load(open(yaml_path))
        prepare_cfg = mod_cfg.get("prepare", mod_cfg.get("data", {}).get("prepare", {}))
        if isinstance(prepare_cfg, dict):
            extra_args = prepare_cfg.get("args", {})
            if isinstance(extra_args, dict):
                for k, v in extra_args.items():
                    args.extend([str(k), str(v)])

    # Pass chat template as --format only if the script accepts it
    # Check by reading the script for --format in argparse
    if chat_template and chat_template != "auto":
        script_text = prepare_script.read_text()
        if "--format" in script_text or "-f" in script_text:
            args.extend(["--format", chat_template])

    result = subprocess.run(
        args,
        cwd=str(module_dir),
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        log(f"  ERROR: prepare.py failed for {module_name}")
        log(f"  stderr: {result.stderr[:500]}")
        return None

    # Print prepare output
    for line in result.stdout.strip().split("\n"):
        log(f"    {line}")

    # Find the output file
    data_dir = module_dir / "data"
    if data_dir.exists():
        jsonl_files = list(data_dir.glob("*.jsonl"))
        if jsonl_files:
            return jsonl_files[0]

    log(f"  WARNING: No data file produced by prepare.py")
    return None


def load_examples(data_path, module_name, weight=1.0, licenses_allowed=None):
    """Load examples from a JSONL file."""
    examples = []
    filtered_license = 0

    with open(data_path) as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)

            # License filtering
            if licenses_allowed:
                ex_license = ex.get("license", "unknown")
                if ex_license not in licenses_allowed and "unknown" not in licenses_allowed:
                    filtered_license += 1
                    continue

            # Add compose metadata
            ex["_module"] = module_name
            ex["_weight"] = weight
            examples.append(ex)

    if filtered_license > 0:
        log(f"  Filtered {filtered_license} examples by license")

    return examples


def apply_weights(all_examples):
    """Apply module weights by repeating/sampling examples."""
    # Group by module
    by_module = {}
    for ex in all_examples:
        module = ex["_module"]
        by_module.setdefault(module, []).append(ex)

    weighted = []
    for module, examples in by_module.items():
        weight = examples[0]["_weight"]
        if weight == 1.0:
            weighted.extend(examples)
        elif weight > 1.0:
            # Repeat examples (with fractional handling)
            full_repeats = int(weight)
            for _ in range(full_repeats):
                weighted.extend(examples)
            # Fractional part: sample proportionally
            frac = weight - full_repeats
            if frac > 0:
                n_extra = int(len(examples) * frac)
                weighted.extend(random.sample(examples, min(n_extra, len(examples))))
        else:
            # weight < 1.0: sample subset
            n_keep = max(1, int(len(examples) * weight))
            weighted.extend(random.sample(examples, n_keep))

        log(f"  {module}: {len(examples)} examples × {weight} weight = {sum(1 for e in weighted if e['_module'] == module)}")

    return weighted


def compose(config_path, output=None, dry_run=False):
    """Main compose pipeline."""
    config = parse_config(config_path)

    log("=" * 60)
    log("TEAPOT COMPOSE")
    log(f"Config: {config_path}")
    log(f"Base model: {config['base_model'] or 'not specified'}")
    log(f"Modules: {len(config['modules'])} configured")
    log(f"Chat template: {config['chat_template']}")
    log("=" * 60)

    enabled_modules = {k: v for k, v in config["modules"].items() if v["enabled"]}
    if not enabled_modules:
        log("ERROR: No modules enabled!")
        sys.exit(1)

    log(f"\nEnabled modules:")
    for name, cfg in enabled_modules.items():
        log(f"  {name}: weight={cfg['weight']}")

    # Phase 1: Prepare
    log("\n--- PREPARE ---")
    module_data = {}

    for module_name, module_cfg in enabled_modules.items():
        log(f"\n[{module_name}]")
        module_dir, yaml_path = find_module(module_name)
        if not module_dir:
            continue

        if dry_run:
            log(f"  (dry run — skipping prepare)")
            continue

        data_path = run_prepare(module_dir, module_name, config["chat_template"])
        if data_path:
            module_data[module_name] = {
                "path": data_path,
                "weight": module_cfg["weight"],
            }

    if dry_run:
        log("\n--- DRY RUN COMPLETE ---")
        return

    # Phase 2: Load & filter
    log("\n--- LOAD ---")
    all_examples = []

    for module_name, data_info in module_data.items():
        examples = load_examples(
            data_info["path"],
            module_name,
            weight=data_info["weight"],
            licenses_allowed=config["licenses_allowed"],
        )
        log(f"  {module_name}: {len(examples)} examples loaded")
        all_examples.extend(examples)

    log(f"\nTotal raw examples: {len(all_examples)}")

    # Phase 3: Weight & merge
    log("\n--- WEIGHT ---")
    random.seed(config["seed"])
    weighted = apply_weights(all_examples)
    random.shuffle(weighted)
    log(f"Total after weighting: {len(weighted)}")

    # Phase 4: Output
    out_path = Path(output) if output else Path(config["output"])
    log(f"\n--- OUTPUT ---")

    # Strip compose metadata before writing
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in weighted:
            # Keep only the fields the trainer needs
            clean = {
                "id": ex.get("id", ""),
                "conversations": ex.get("conversations", []),
            }
            # Optionally keep metadata
            if "category" in ex:
                clean["category"] = ex["category"]
            if "source" in ex:
                clean["source"] = ex["source"]
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")

    log(f"Wrote {len(weighted)} examples to {out_path}")

    # Phase 5: Manifest
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "config": str(config_path),
        "base_model": config["base_model"],
        "chat_template": config["chat_template"],
        "seed": config["seed"],
        "total_examples": len(weighted),
        "modules": {},
    }

    for module_name, data_info in module_data.items():
        count = sum(1 for e in weighted if e.get("_module") == module_name)
        manifest["modules"][module_name] = {
            "source": str(data_info["path"]),
            "weight": data_info["weight"],
            "examples_raw": sum(1 for e in all_examples if e.get("_module") == module_name),
            "examples_weighted": count,
            "integrity": "sha256:" + hashlib.sha256(
                open(data_info["path"], "rb").read()
            ).hexdigest(),
        }

    # Output hash for reproducibility
    manifest["output_hash"] = "sha256:" + hashlib.sha256(
        open(out_path, "rb").read()
    ).hexdigest()

    manifest_path = out_path.with_suffix(".manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    log(f"Manifest: {manifest_path}")

    log("\n" + "=" * 60)
    log("COMPOSE COMPLETE")
    log(f"  Output: {out_path} ({len(weighted)} examples)")
    log(f"  Manifest: {manifest_path}")
    by_module = Counter(e.get("_module") for e in weighted)
    for mod, cnt in by_module.most_common():
        log(f"    {mod}: {cnt}")
    log("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Teapot compose — merge module data")
    parser.add_argument("config", help="Config file path")
    parser.add_argument("--output", "-o", help="Output JSONL file (overrides config)")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without running")
    parser.add_argument("--lock", action="store_true", help="Generate teapot.lock after compose")
    args = parser.parse_args()

    compose(args.config, output=args.output, dry_run=args.dry_run)

    if args.lock and not args.dry_run:
        from lockfile import generate_lock
        out_path = Path(args.output) if args.output else Path(parse_config(args.config)["output"])
        manifest_path = out_path.with_suffix(".manifest.json")
        lock_path = Path("teapot.lock")
        generate_lock(manifest_path, lock_path)
        log(f"Lock: {lock_path}")


if __name__ == "__main__":
    main()
