#!/usr/bin/env python3
"""Launch a vLLM or Ollama server from a YAML config file."""

import argparse
import glob
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── vLLM backend ───────────────────────────────────────────────────────────────

def _vllm_cmd(cfg: dict) -> list[str]:
    model_cfg = cfg.get("model", {})
    server_cfg = cfg.get("server", {})
    vllm_cfg = cfg.get("vllm", {})

    model = model_cfg.get("local_path") or model_cfg.get("name")
    if not model:
        raise ValueError("config.yaml must set model.name or model.local_path")

    cmd: list[str] = ["vllm", "serve", model]

    cmd += ["--host", str(server_cfg.get("host", "0.0.0.0"))]
    cmd += ["--port", str(server_cfg.get("port", 8000))]

    if server_cfg.get("api_key"):
        cmd += ["--api-key", server_cfg["api_key"]]
    if server_cfg.get("served_model_name"):
        cmd += ["--served-model-name", server_cfg["served_model_name"]]
    if model_cfg.get("revision"):
        cmd += ["--revision", model_cfg["revision"]]
    if model_cfg.get("tokenizer"):
        cmd += ["--tokenizer", model_cfg["tokenizer"]]

    int_args = {
        "tensor_parallel_size": "--tensor-parallel-size",
        "pipeline_parallel_size": "--pipeline-parallel-size",
        "max_model_len": "--max-model-len",
        "max_num_seqs": "--max-num-seqs",
    }
    float_args = {
        "gpu_memory_utilization": "--gpu-memory-utilization",
    }
    str_args = {
        "dtype": "--dtype",
        "quantization": "--quantization",
        "tokenizer_mode": "--tokenizer-mode",
    }
    bool_flags = {
        "enable_prefix_caching": "--enable-prefix-caching",
    }

    for key, flag in int_args.items():
        val = vllm_cfg.get(key)
        if val is not None:
            cmd += [flag, str(val)]
    for key, flag in float_args.items():
        val = vllm_cfg.get(key)
        if val is not None:
            cmd += [flag, str(val)]
    for key, flag in str_args.items():
        val = vllm_cfg.get(key)
        if val is not None:
            cmd += [flag, str(val)]
    for key, flag in bool_flags.items():
        if vllm_cfg.get(key) is True:
            cmd.append(flag)

    return cmd


def run_vllm(cfg: dict, dry_run: bool) -> None:
    cmd = _vllm_cmd(cfg)
    print("Command:", " ".join(cmd))
    if dry_run:
        return
    os.execvp(cmd[0], cmd)


# ── Ollama backend ─────────────────────────────────────────────────────────────

def _build_modelfile(cfg: dict) -> tuple[str, str]:
    """Return (model_name, modelfile_content)."""
    model_cfg = cfg.get("model", {})
    ollama_cfg = cfg.get("ollama", {})

    model_name = ollama_cfg.get("model_name")
    if not model_name:
        raise ValueError("ollama.model_name must be set in config.yaml")

    local_path = model_cfg.get("local_path")
    if not local_path:
        raise ValueError("model.local_path must be set to use the ollama backend")

    gguf_glob = ollama_cfg.get("gguf_glob", "*.gguf")
    pattern = str(Path(local_path) / gguf_glob)
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No GGUF file found matching '{pattern}'.\n"
            "Download GGUF weights with:  ./download.sh --gguf-only"
        )

    gguf_path = str(Path(matches[0]).resolve())

    lines = [f"FROM {gguf_path}"]

    system = ollama_cfg.get("system")
    if system:
        # Escape any triple-quotes inside the system prompt
        system = system.replace('"""', '\\"\\"\\"')
        lines.append(f'SYSTEM """{system}"""')

    for key, val in (ollama_cfg.get("parameters") or {}).items():
        lines.append(f"PARAMETER {key} {val}")

    return model_name, "\n".join(lines) + "\n"


def run_ollama(cfg: dict, dry_run: bool) -> None:
    model_name, modelfile = _build_modelfile(cfg)

    print("── Modelfile " + "─" * 50)
    print(modelfile)
    print("─" * 62)

    if dry_run:
        print(f"Would run: ollama create {model_name} -f <modelfile>")
        print("Would run: ollama serve")
        return

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".Modelfile", delete=False
    ) as f:
        f.write(modelfile)
        modelfile_path = f.name

    try:
        print(f"Creating ollama model '{model_name}'...")
        subprocess.run(
            ["ollama", "create", model_name, "-f", modelfile_path],
            check=True,
        )
    finally:
        os.unlink(modelfile_path)

    print("Starting ollama serve...")
    os.execvp("ollama", ["ollama", "serve"])


# ── entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve a model with vLLM or Ollama using a YAML config"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        metavar="FILE",
        help="Path to config YAML (default: config.yaml)",
    )
    parser.add_argument(
        "--backend",
        choices=["vllm", "ollama"],
        default="vllm",
        help="Inference backend (default: vllm)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be executed without running it",
    )
    args = parser.parse_args()

    if not Path(args.config).exists():
        sys.exit(f"Config file not found: {args.config}")

    cfg = load_config(args.config)

    if args.backend == "vllm":
        run_vllm(cfg, args.dry_run)
    else:
        run_ollama(cfg, args.dry_run)


if __name__ == "__main__":
    main()
