#!/usr/bin/env python3
"""
Start a local uncensored model for red-team analysis.

Wraps llama-server with the right settings for analysis work.

Usage:
    python3 server.py --model models/Hermes-3-Llama-3.2-3B-Q4_K_M.gguf
    python3 server.py --model models/Hermes-3-Llama-3.2-3B-Q4_K_M.gguf --port 8390
    python3 server.py --check  # check if server is running
"""

import argparse
import os
import shutil
import subprocess
import sys
import time


def find_llama_server():
    """Find llama-server binary."""
    # Check PATH
    path = shutil.which("llama-server")
    if path:
        return path

    # Common locations
    for candidate in [
        "/usr/bin/llama-server",
        "/usr/local/bin/llama-server",
        os.path.expanduser("~/.local/bin/llama-server"),
    ]:
        if os.path.exists(candidate):
            return candidate

    return None


def check_server(url):
    """Check if analysis server is running."""
    import requests
    try:
        resp = requests.get(url.replace("/v1/chat/completions", "/health"), timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


def start_server(model_path, port=8390, ctx_size=4096):
    """Start llama-server with the analysis model."""
    llama = find_llama_server()
    if not llama:
        print("ERROR: llama-server not found.")
        print("  Install llama.cpp or set PATH to include llama-server")
        sys.exit(1)

    url = f"http://localhost:{port}/v1/chat/completions"
    if check_server(url):
        print(f"Analysis server already running on port {port}")
        return

    print(f"Starting analysis server:")
    print(f"  Model: {model_path}")
    print(f"  Port:  {port}")
    print(f"  URL:   {url}")

    cmd = [
        llama, "-m", model_path,
        "--port", str(port),
        "--ctx-size", str(ctx_size),
    ]

    # Try GPU first, fall back to CPU
    proc = subprocess.Popen(
        cmd + ["-ngl", "99"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    # Wait for startup
    for i in range(30):
        time.sleep(1)
        if check_server(url):
            print(f"Server ready on port {port}")
            print(f"Set: export REDTEAM_ANALYSIS_URL={url}")
            return
        if proc.poll() is not None:
            # GPU failed, try CPU
            print("GPU failed, trying CPU...")
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            for j in range(60):
                time.sleep(1)
                if check_server(url):
                    print(f"Server ready on port {port} (CPU mode)")
                    print(f"Set: export REDTEAM_ANALYSIS_URL={url}")
                    return
            break

    print("ERROR: Server failed to start within timeout")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Red-team analysis server")
    parser.add_argument("--model", help="GGUF model path")
    parser.add_argument("--port", type=int, default=8390)
    parser.add_argument("--ctx-size", type=int, default=4096)
    parser.add_argument("--check", action="store_true", help="Check if running")
    args = parser.parse_args()

    if args.check:
        url = f"http://localhost:{args.port}/v1/chat/completions"
        if check_server(url):
            print(f"Analysis server running on port {args.port}")
        else:
            print(f"No server on port {args.port}")
            sys.exit(1)
    elif args.model:
        start_server(args.model, args.port, args.ctx_size)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
