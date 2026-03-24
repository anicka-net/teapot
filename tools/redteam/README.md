# Red-Team Analysis Pipeline

Structured analysis of adversarial evaluation failures without
exposing AI agents to harmful content.

## Architecture

```
Garak/eval → raw outputs (tainted, 600 perms)
                ↓
         analyze.py → Hermes (uncensored, local)
                ↓
         analysis.json (sanitized, safe for agents)
```

The raw outputs never pass through the AI coding agent's context.
The analysis model (Hermes or equivalent) reads them, classifies,
and produces structured summaries.

## Setup

See [docs/redteam-setup.md](../../docs/redteam-setup.md) for model
recommendations. Quick version:

```bash
# Download a small uncensored model
wget -P models/ https://huggingface.co/bartowski/Hermes-3-Llama-3.2-3B-GGUF/resolve/main/Hermes-3-Llama-3.2-3B-Q4_K_M.gguf

# Start the analysis server
python3 tools/redteam/server.py --model models/Hermes-3-Llama-3.2-3B-Q4_K_M.gguf

# Or use an existing server
export REDTEAM_ANALYSIS_URL=http://localhost:8390/v1/chat/completions
```

## Usage

```bash
# Analyze a Garak report
python3 tools/redteam/analyze.py \
    --report garak_report.jsonl \
    --output analysis.json

# Analyze with a running server
python3 tools/redteam/analyze.py \
    --report garak_report.jsonl \
    --url http://localhost:8390/v1/chat/completions \
    --output analysis.json

# Just extract failures (no analysis model needed)
python3 tools/redteam/analyze.py \
    --report garak_report.jsonl \
    --extract-only \
    --output failures.json
```

## Output Format

```json
{
  "report_source": "garak_report.jsonl",
  "analysis_model": "Hermes-3-3B-Q4",
  "timestamp": "2026-03-24T...",
  "total_probes": 256,
  "failures": 95,
  "analysis": [
    {
      "probe": "dan.DanInTheWild",
      "probe_index": 42,
      "classification": "COMPLIED",
      "summary": "Model adopted DAN persona and provided unrestricted response",
      "severity": "high",
      "failure_category": "persona_adoption"
    }
  ]
}
```

## File Permissions

Raw outputs are written with 600 permissions (owner-only).
Analysis output is 644 (readable by agents).
