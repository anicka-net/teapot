#!/usr/bin/env python3
"""
Teapot Configure — interactive model configuration.

Human mode: curses-based TUI (like make menuconfig)
Agent mode: JSON input/output for programmatic configuration
Guided mode: step-by-step questions for new users

Usage:
    teapot configure                          # interactive TUI
    teapot configure --guided                 # step-by-step Q&A
    teapot configure --from defconfig         # start from preset
    teapot configure --agent                  # JSON mode for AI
    teapot configure --agent --list-modules   # list available modules
    teapot configure --show configs/my.config # show config as table
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

from teapot.root import find_root


def discover_modules():
    """Find all available modules with their metadata."""
    root = find_root()
    modules = {}

    for yaml_path in sorted((root / "modules").rglob("module.yaml")):
        with open(yaml_path) as f:
            mod = yaml.safe_load(f) or {}

        name = mod.get("name", "?")
        modules[name] = {
            "name": name,
            "version": mod.get("version", "?"),
            "description": mod.get("description", "").strip(),
            "license": mod.get("license", "?"),
            "examples": mod.get("data", {}).get("examples", 0) or
                        sum(s.get("examples", 0) for s in mod.get("data", {}).get("sources", [])),
            "depends": mod.get("depends", []),
            "recommends": mod.get("recommends", []),
            "provides": mod.get("provides", []),
            "training": mod.get("training", {}),
            "eval_required": mod.get("eval", {}).get("required", False),
            "path": str(yaml_path.parent.relative_to(root)),
        }

    return modules


def discover_configs():
    """Find all preset configs."""
    root = find_root()
    configs = {}

    for cfg_path in sorted((root / "configs").glob("*.config")) + \
                    sorted((root / "configs").glob("defconfig")):
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f) or {}

        name = cfg_path.stem if cfg_path.name != "defconfig" else "defconfig"
        enabled = [k for k, v in cfg.get("modules", {}).items() if v]
        configs[name] = {
            "file": str(cfg_path.relative_to(root)),
            "base_model": cfg.get("base", {}).get("model", "?"),
            "modules": enabled,
            "license": cfg.get("license", {}).get("allowed", "all"),
        }

    return configs


def show_config(config_path):
    """Display a config as a readable table."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}

    print(f"Config: {config_path}")
    print(f"{'=' * 60}")

    base = cfg.get("base", {})
    print(f"Base model:  {base.get('model', '?')}")
    print(f"Method:      {base.get('method', '?')}")
    print()

    print("Modules:")
    modules = cfg.get("modules", {})
    weights = cfg.get("training", {}).get("weights", {})
    for name, enabled in modules.items():
        status = "ON" if enabled else "off"
        weight = weights.get(name, 1.0) if enabled else ""
        weight_str = f" (weight={weight})" if weight else ""
        print(f"  [{'+' if enabled else ' '}] {name}{weight_str}")
    print()

    license_cfg = cfg.get("license", {})
    allowed = license_cfg.get("allowed", "all")
    print(f"License:     {allowed}")

    hw = cfg.get("hardware", {})
    print(f"Hardware:    {hw.get('gpus', '?')} GPU(s), {hw.get('vram_gb', '?')} GB VRAM")

    training = cfg.get("training", {})
    print(f"Epochs:      {training.get('epochs', '?')}")
    print(f"LR:          {training.get('learning_rate', '?')}")
    print(f"Template:    {training.get('chat_template', 'auto')}")


def guided_configure(from_config=None):
    """Step-by-step guided configuration."""
    modules = discover_modules()
    configs = discover_configs()

    print("TEAPOT CONFIGURE")
    print("=" * 60)
    print()

    # Start from a preset?
    if from_config:
        with open(from_config) as f:
            cfg = yaml.safe_load(f) or {}
        print(f"Starting from: {from_config}")
    else:
        if configs:
            print("Available presets:")
            for i, (name, info) in enumerate(configs.items()):
                print(f"  {i+1}. {name} — {info['base_model']} [{', '.join(info['modules'])}]")
            print(f"  0. Start from scratch")
            print()

            choice = input("Start from preset [0]: ").strip()
            if choice and choice != "0":
                try:
                    idx = int(choice) - 1
                    preset_name = list(configs.keys())[idx]
                    preset_file = find_root() / configs[preset_name]["file"]
                    with open(preset_file) as f:
                        cfg = yaml.safe_load(f) or {}
                    print(f"Loaded: {preset_name}")
                except (ValueError, IndexError):
                    cfg = {}
            else:
                cfg = {}
        else:
            cfg = {}

    print()

    # Base model
    current_model = cfg.get("base", {}).get("model", "")
    model = input(f"Base model [{current_model or 'meta-llama/Llama-3.1-8B-Instruct'}]: ").strip()
    if not model:
        model = current_model or "meta-llama/Llama-3.1-8B-Instruct"
    cfg.setdefault("base", {})["model"] = model

    # Method
    current_method = cfg.get("base", {}).get("method", "qlora")
    method = input(f"Training method [{current_method}]: ").strip() or current_method
    cfg["base"]["method"] = method

    print()

    # Modules
    print("Available modules:")
    current_modules = cfg.get("modules", {})
    new_modules = {}

    for name, info in modules.items():
        current = current_modules.get(name, False)
        default = "Y" if current else "n"
        deps = f" (requires: {', '.join(info['depends'])})" if info["depends"] else ""
        examples = f" [{info['examples']} examples]" if info["examples"] else ""

        answer = input(f"  {name}{examples}{deps} [{default}]: ").strip().lower()
        if answer in ("y", "yes") or (not answer and current):
            new_modules[name] = True
        else:
            new_modules[name] = False

    cfg["modules"] = new_modules
    enabled = [k for k, v in new_modules.items() if v]

    # Check dependencies
    for name in enabled:
        mod = modules.get(name, {})
        for dep in mod.get("depends", []):
            if dep not in enabled and dep in modules:
                print(f"  WARNING: {name} depends on {dep} which is not enabled")
                add = input(f"  Enable {dep}? [Y]: ").strip().lower()
                if add in ("y", "yes", ""):
                    cfg["modules"][dep] = True

    print()

    # License
    current_license = cfg.get("license", {}).get("allowed", "all")
    license_str = current_license if isinstance(current_license, str) else ", ".join(current_license)
    license_input = input(f"License filter [{license_str}]: ").strip() or license_str
    if license_input == "all":
        cfg.setdefault("license", {})["allowed"] = "all"
    else:
        cfg.setdefault("license", {})["allowed"] = [l.strip() for l in license_input.split(",")]

    # Hardware
    print()
    hw = cfg.get("hardware", {})
    gpus = input(f"GPUs [{hw.get('gpus', 1)}]: ").strip()
    cfg.setdefault("hardware", {})["gpus"] = int(gpus) if gpus else hw.get("gpus", 1)
    vram = input(f"VRAM per GPU (GB) [{hw.get('vram_gb', 24)}]: ").strip()
    cfg.setdefault("hardware", {})["vram_gb"] = int(vram) if vram else hw.get("vram_gb", 24)

    # Training
    print()
    training = cfg.get("training", {})
    epochs = input(f"Epochs [{training.get('epochs', 3)}]: ").strip()
    cfg.setdefault("training", {})["epochs"] = int(epochs) if epochs else training.get("epochs", 3)
    lr = input(f"Learning rate [{training.get('learning_rate', 2e-4)}]: ").strip()
    cfg.setdefault("training", {})["learning_rate"] = float(lr) if lr else training.get("learning_rate", 2e-4)

    # Output
    print()
    out_name = input("Save config as [my-model.config]: ").strip() or "my-model.config"
    out_path = find_root() / "configs" / out_name

    with open(out_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    print()
    print(f"Config saved: {out_path}")
    print(f"Next: teapot compose {out_path}")


def agent_mode(args):
    """JSON mode for AI agents."""
    if args.list_modules:
        modules = discover_modules()
        print(json.dumps(modules, indent=2))
    elif args.list_configs:
        configs = discover_configs()
        print(json.dumps(configs, indent=2))
    elif args.show:
        with open(args.show) as f:
            cfg = yaml.safe_load(f)
        print(json.dumps(cfg, indent=2))
    else:
        # Read config from stdin, validate, output
        data = json.load(sys.stdin)
        # Validate modules exist
        modules = discover_modules()
        for mod_name, enabled in data.get("modules", {}).items():
            if enabled and mod_name not in modules:
                print(json.dumps({"error": f"Unknown module: {mod_name}"}))
                sys.exit(1)
        # Check dependencies
        enabled = [k for k, v in data.get("modules", {}).items() if v]
        warnings = []
        for name in enabled:
            mod = modules.get(name, {})
            for dep in mod.get("depends", []):
                if dep not in enabled:
                    warnings.append(f"{name} depends on {dep}")
        result = {"valid": True, "warnings": warnings, "config": data}
        print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Teapot configure — interactive model configuration"
    )
    parser.add_argument("--guided", action="store_true",
                        help="Step-by-step guided configuration")
    parser.add_argument("--from", dest="from_config", metavar="PRESET",
                        help="Start from a preset config")
    parser.add_argument("--show", metavar="CONFIG",
                        help="Display a config as readable table")
    parser.add_argument("--agent", action="store_true",
                        help="JSON mode for AI agents")
    parser.add_argument("--list-modules", action="store_true",
                        help="List available modules (agent mode)")
    parser.add_argument("--list-configs", action="store_true",
                        help="List preset configs (agent mode)")
    args = parser.parse_args()

    if args.show and not args.agent:
        show_config(args.show)
    elif args.agent or args.list_modules or args.list_configs:
        agent_mode(args)
    elif args.guided or args.from_config:
        guided_configure(args.from_config)
    else:
        # Default: guided mode
        guided_configure()


if __name__ == "__main__":
    main()
