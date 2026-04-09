#!/usr/bin/env python3
"""Launch a vLLM OpenAI-compatible server from a YAML config file."""

import argparse
import os
import sys
from pathlib import Path

import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_cmd(cfg: dict) -> list[str]:
    model_cfg = cfg.get("model", {})
    server_cfg = cfg.get("server", {})
    vllm_cfg = cfg.get("vllm", {})

    # Prefer an explicit local path; fall back to the HF repo ID.
    model = model_cfg.get("local_path") or model_cfg.get("name")
    if not model:
        raise ValueError("config.yaml must set model.name or model.local_path")

    cmd: list[str] = ["vllm", "serve", model]

    # ── server ──────────────────────────────────────────────────────────────
    cmd += ["--host", str(server_cfg.get("host", "0.0.0.0"))]
    cmd += ["--port", str(server_cfg.get("port", 8000))]

    if server_cfg.get("api_key"):
        cmd += ["--api-key", server_cfg["api_key"]]

    if server_cfg.get("served_model_name"):
        cmd += ["--served-model-name", server_cfg["served_model_name"]]

    # ── model identity ───────────────────────────────────────────────────────
    if model_cfg.get("revision"):
        cmd += ["--revision", model_cfg["revision"]]

    if model_cfg.get("tokenizer"):
        cmd += ["--tokenizer", model_cfg["tokenizer"]]

    # ── vllm engine ──────────────────────────────────────────────────────────
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve a model with vLLM using a YAML config"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        metavar="FILE",
        help="Path to config YAML (default: config.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the vllm command without executing it",
    )
    args = parser.parse_args()

    if not Path(args.config).exists():
        sys.exit(f"Config file not found: {args.config}")

    cfg = load_config(args.config)
    cmd = build_cmd(cfg)

    print("Command:", " ".join(cmd))

    if args.dry_run:
        return

    # Replace the current process so signals (SIGINT, SIGTERM) go straight
    # to vllm instead of being caught by a Python wrapper.
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
