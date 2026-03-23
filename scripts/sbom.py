#!/usr/bin/env python3
"""
Teapot SBOM — generate SPDX 3.0 AI Profile from compose manifest.

The manifest is the source of truth. The SBOM is a view of it in
the SPDX format that regulators and procurement systems expect.

Usage:
    python3 scripts/sbom.py train-apertus-70b-secular.manifest.json
    python3 scripts/sbom.py manifest.json --output model.sbom.json
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import yaml

TEAPOT_ROOT = Path(__file__).resolve().parents[1]


def load_module_metadata(module_name):
    """Load metadata from a module's module.yaml."""
    parts = module_name.split("/")
    yaml_path = TEAPOT_ROOT / "modules" / "/".join(parts) / "module.yaml"
    if not yaml_path.exists():
        return {}
    with open(yaml_path) as f:
        return yaml.safe_load(f) or {}


def generate_sbom(manifest_path, output=None):
    """Generate SPDX 3.0 AI Profile SBOM from manifest."""
    manifest = json.loads(Path(manifest_path).read_text())

    sbom = {
        "spdxVersion": "SPDX-3.0",
        "dataLicense": "CC0-1.0",
        "creationInfo": {
            "created": datetime.now().isoformat(),
            "createdBy": ["Tool: teapot"],
            "specVersion": "3.0",
        },
        "elements": [],
    }

    # Training run element
    training_run = {
        "type": "ai:TrainedModel",
        "name": f"teapot-{Path(manifest.get('config', 'unknown')).stem}",
        "baseModel": manifest.get("base_model", "unknown"),
        "chatTemplate": manifest.get("chat_template", "auto"),
        "informationAboutTraining": {
            "totalExamples": manifest.get("total_examples", 0),
            "seed": manifest.get("seed", 42),
            "composeTimestamp": manifest.get("timestamp", ""),
            "outputHash": manifest.get("output_hash", ""),
        },
        "trainingDatasets": [],
    }

    # License summary
    all_licenses = set()

    # Dataset elements
    for module_name, module_info in manifest.get("modules", {}).items():
        mod_meta = load_module_metadata(module_name)

        # Determine license
        license_info = mod_meta.get("data", {}).get("licenses", {})
        mod_license = license_info.get("default", mod_meta.get("license", "unknown"))
        all_licenses.add(mod_license)

        dataset = {
            "type": "ai:TrainingDataset",
            "name": module_name,
            "version": mod_meta.get("version", "unknown"),
            "description": mod_meta.get("description", "").strip(),
            "suppliedBy": mod_meta.get("maintainer", "unknown"),
            "declaredLicense": mod_license,
            "examplesRaw": module_info.get("examples_raw", 0),
            "examplesWeighted": module_info.get("examples_weighted", 0),
            "weight": module_info.get("weight", 1.0),
            "contentHash": module_info.get("integrity", ""),
            "sourcePath": module_info.get("source", ""),
        }

        # Per-example license info if mixed
        if mod_license == "mixed" and "known_licenses" in license_info:
            dataset["knownLicenses"] = license_info["known_licenses"]

        sbom["elements"].append(dataset)
        training_run["trainingDatasets"].append(module_name)

    # Add training run as first element
    training_run["licenseSummary"] = sorted(all_licenses)
    sbom["elements"].insert(0, training_run)

    # Output
    out_path = Path(output) if output else Path(manifest_path).with_suffix(".sbom.json")
    with open(out_path, "w") as f:
        json.dump(sbom, f, indent=2)

    print(f"Generated SBOM: {out_path}")
    print(f"  Format: SPDX 3.0 AI Profile")
    print(f"  Elements: {len(sbom['elements'])} (1 model + {len(sbom['elements'])-1} datasets)")
    print(f"  Licenses: {', '.join(sorted(all_licenses))}")


def main():
    parser = argparse.ArgumentParser(description="Generate SPDX 3.0 SBOM from manifest")
    parser.add_argument("manifest", help="Compose manifest JSON path")
    parser.add_argument("--output", "-o", help="Output SBOM file")
    args = parser.parse_args()

    generate_sbom(args.manifest, args.output)


if __name__ == "__main__":
    main()
