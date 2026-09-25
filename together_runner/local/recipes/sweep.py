#!/usr/bin/env python3
"""Drive an official InferenceX sweep matrix on a local box (NVIDIA or AMD).

Generates the same config matrix CI would run (via
utils/matrix_logic/generate_sweep_configs.py against the vendor master config),
then runs each entry through launch_local.sh and utils/process_result.py,
collecting agg_*.json into together_runner/results/<hw>/.

Examples:
    # AMD MI350X, using the mi355x recipes (same gfx950 silicon)
    python3 sweep.py --model-prefix gptoss --framework vllm \
        --recipe-runner mi355x --hw mi350x --tp 1 --conc 4 8 16

    # NVIDIA B200
    python3 sweep.py --model-prefix gptoss --framework vllm \
        --recipe-runner b200 --hw b200
"""
import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent          # together_runner/local/recipes
TR_ROOT = HERE.parent.parent                    # together_runner
REPO = TR_ROOT.parent                           # repo root
LAUNCHER = HERE / "launch_local.sh"
GENERATOR = REPO / "utils" / "matrix_logic" / "generate_sweep_configs.py"
RESULTS_ROOT = TR_ROOT / "results"

# The matrix generator imports pydantic + yaml, which a bare system python
# usually lacks. Find an interpreter that has them rather than failing deep
# inside a subprocess with an opaque CalledProcessError.
_GEN_DEPS = "import pydantic, yaml"


def _can_generate(py: str) -> bool:
    try:
        return subprocess.run([py, "-c", _GEN_DEPS],
                              capture_output=True).returncode == 0
    except OSError:
        return False


def resolve_gen_python() -> str:
    explicit = os.environ.get("GEN_PYTHON")
    if explicit:
        if not _can_generate(explicit):
            raise SystemExit(f"GEN_PYTHON={explicit} cannot import pydantic+yaml")
        return explicit
    candidates = [sys.executable,
                  str(REPO / ".venv" / "bin" / "python"),
                  str(Path.home() / ".venv-inferencex" / "bin" / "python")]
    for c in candidates:
        if c and Path(c).exists() and _can_generate(c):
            return c
    raise SystemExit(
        "no python with pydantic+yaml found (needed by "
        "utils/matrix_logic/generate_sweep_configs.py).\n"
        "Fix with either:\n"
        f"  python3 -m venv {REPO}/.venv && {REPO}/.venv/bin/pip install pydantic pyyaml\n"
        "  GEN_PYTHON=/path/to/python python3 sweep.py ...\n"
        f"tried: {', '.join(candidates)}")


def master_config(recipe_runner: str) -> Path:
    """AMD runners live in amd-master.yaml, everything else in nvidia-master.yaml."""
    amd = recipe_runner.startswith("mi")
    name = "amd-master.yaml" if amd else "nvidia-master.yaml"
    return REPO / ".github" / "configs" / name


def detect_hw() -> str:
    """Best-effort hardware tag from the local GPU (mi350x, b200, h200, ...)."""
    import re
    import shutil
    if Path("/dev/kfd").exists() and shutil.which("rocm-smi"):
        out = subprocess.run(["rocm-smi", "--showproductname"],
                             capture_output=True, text=True).stdout
        m = re.search(r"MI(\d+)X?", out, re.I)
        if m:
            return f"mi{m.group(1)}x".lower()
    if shutil.which("nvidia-smi"):
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True).stdout
        m = re.search(r"\b([BHA])(\d{3})\b", out)
        if m:
            return f"{m.group(1)}{m.group(2)}".lower()
    raise SystemExit("could not detect hardware; pass --hw")


def generate_matrix(args) -> list[dict]:
    cmd = [resolve_gen_python(), str(GENERATOR), "full-sweep",
           "--config-files", str(master_config(args.recipe_runner)),
           "--model-prefix", args.model_prefix,
           "--framework", args.framework,
           "--runner-type", args.recipe_runner,
           "--seq-lens", args.seq_lens]
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"matrix generation failed:\n{proc.stderr.strip()}")
    out = proc.stdout
    matrix = json.loads(out.strip().splitlines()[-1])
    if args.tp:
        matrix = [e for e in matrix if e["tp"] in args.tp]
    if args.conc:
        matrix = [e for e in matrix if e["conc"] in args.conc]
    return matrix


def result_filename(e: dict, runner: str) -> str:
    return (f'{e["exp-name"]}_{e["precision"]}_{e["framework"]}'
            f'_tp{e["tp"]}-ep{e["ep"]}-dpa{str(e["dp-attn"]).lower()}'
            f'_disagg-{str(e["disagg"]).lower()}_spec-{e["spec-decoding"]}'
            f'_conc{e["conc"]}_{runner}')


def build_env(e: dict, args, work_dir: Path, rf: str) -> dict:
    """The env contract launch_local.sh and process_result.py both read.

    Deliberately the same variable names .github/workflows/benchmark-tmpl.yml
    exports, so the recipes cannot tell they are running outside CI.
    """
    env = dict(os.environ)
    env.update({
        "MODEL": args.model_path or e["model"],
        "TP": str(e["tp"]), "EP_SIZE": str(e["ep"]),
        "DP_ATTENTION": str(e["dp-attn"]).lower(), "CONC": str(e["conc"]),
        "ISL": str(e["isl"]), "OSL": str(e["osl"]),
        "MAX_MODEL_LEN": str(e["max-model-len"]),
        "RANDOM_RANGE_RATIO": "0.8", "RESULT_FILENAME": rf,
        "IMAGE": e["image"], "FRAMEWORK": e["framework"],
        "PRECISION": e["precision"], "EXP_NAME": e["exp-name"],
        "SPEC_DECODING": e["spec-decoding"], "DISAGG": str(e["disagg"]).lower(),
        "MODEL_PREFIX": e["model-prefix"], "RUNNER_TYPE": args.hw,
        "RECIPE_RUNNER": args.recipe_runner,
        "WORK_DIR": str(work_dir), "PORT": str(args.port),
    })
    if args.gpus:
        env["GPUS"] = args.gpus
    return env


def harvest(work_dir: Path, out_dir: Path, rf: str) -> Path:
    """Move this config's outputs out of the shared scratch dir."""
    for name in (f"{rf}.json", f"agg_{rf}.json", "server.log", "gpu_metrics.csv"):
        src = work_dir / name
        if src.exists():
            dst = out_dir / (name if name.startswith(("agg_", rf)) else f"{rf}.{name}")
            shutil.move(str(src), dst)
    return out_dir / f"agg_{rf}.json"


def stamp_provenance(agg_file: Path, e: dict, args) -> dict:
    """process_result.py records none of this, but analysis/ groups results by it."""
    agg = json.loads(agg_file.read_text())
    agg.setdefault("host", socket.gethostname())
    agg.setdefault("cluster", os.environ.get("CLUSTER", "local"))
    agg.setdefault("ts", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    if args.model_path:
        # bench_serving records model_id as whatever was passed to --model, so a
        # local path would land in the schema where the HF id belongs. Restore
        # the canonical id and keep the path for provenance.
        agg["model"] = e["model"]
        agg["model_path"] = args.model_path
    agg_file.write_text(json.dumps(agg, indent=2))
    return agg


def run_one(e: dict, args, work_dir: Path, out_dir: Path) -> dict:
    rf = result_filename(e, args.hw)
    env = build_env(e, args, work_dir, rf)

    print(f"\n{'='*78}\n[sweep] {rf}\n{'='*78}", flush=True)
    log_path = out_dir / f"{rf}.launch.log"
    with open(log_path, "w") as log:
        rc = subprocess.run(["bash", str(LAUNCHER)], env=env, cwd=REPO,
                            stdout=log, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        print(f"[sweep] FAILED (rc={rc}); see {log_path}", flush=True)
        src = work_dir / "server.log"
        if src.exists():
            shutil.copy(src, out_dir / f"{rf}.server.log")
        return {"config": rf, "status": "failed", "rc": rc}

    # process_result.py reads ./$RESULT_FILENAME.json and writes agg_*.json
    subprocess.run([sys.executable, str(REPO / "utils" / "process_result.py")],
                   env=env, cwd=work_dir, check=True)
    agg = stamp_provenance(harvest(work_dir, out_dir, rf), e, args)
    print(f"[sweep] OK  tput/gpu={agg['tput_per_gpu']:.0f} tok/s  "
          f"out/gpu={agg['output_tput_per_gpu']:.0f} tok/s", flush=True)
    return {"config": rf, "status": "ok", "agg": agg}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-prefix", required=True)
    p.add_argument("--framework", required=True)
    p.add_argument("--seq-lens", default="1k1k")
    p.add_argument("--recipe-runner", default=None,
                   help="runner token in the recipe filename (mi355x, b200, h200, ...). "
                        "Defaults to --hw.")
    p.add_argument("--hw", default=None,
                   help="hardware tag recorded in results/baselines. Defaults to the "
                        "detected GPU (e.g. mi350x, b200).")
    p.add_argument("--tp", type=int, nargs="*")
    p.add_argument("--conc", type=int, nargs="*")
    p.add_argument("--port", type=int, default=8710)
    p.add_argument("--gpus", help="ROCR_VISIBLE_DEVICES, e.g. 0,1,2,3")
    p.add_argument("--model-path", help="Serve weights from this in-container "
                   "path instead of letting the recipe run `hf download`. Must live "
                   "under the mounted HF cache so the snapshot's relative blob "
                   "symlinks still resolve.")
    p.add_argument("--work-dir",
                   default=os.environ.get("TR_WORK_DIR", "/tmp/inferencex-work"))
    p.add_argument("--out-dir", default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if not args.hw:
        args.hw = detect_hw()
    if not args.recipe_runner:
        args.recipe_runner = args.hw
    print(f"[sweep] hw={args.hw} recipes={args.recipe_runner} "
          f"config={master_config(args.recipe_runner).name}")

    matrix = generate_matrix(args)
    print(f"[sweep] {len(matrix)} config(s):")
    for e in matrix:
        print(f"  tp={e['tp']} ep={e['ep']} conc={e['conc']} "
              f"{e['isl']}/{e['osl']} {e['image']}")
    if args.dry_run:
        return

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out_dir or (RESULTS_ROOT / args.hw /
                                    f"{args.model_prefix}-{args.framework}-{stamp}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path(args.work_dir)
    print(f"[sweep] results -> {out_dir}")

    summary = [run_one(e, args, work_dir, out_dir) for e in matrix]
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    ok = [s for s in summary if s["status"] == "ok"]
    print(f"\n[sweep] {len(ok)}/{len(summary)} succeeded -> {out_dir}")
    if ok:
        print(f"\n{'tp':>3} {'conc':>5} {'tput/gpu':>10} {'out/gpu':>9} "
              f"{'TTFT p99(s)':>12} {'TPOT med(s)':>12}")
        for s in ok:
            a = s["agg"]
            print(f"{a['tp']:>3} {a['conc']:>5} {a['tput_per_gpu']:>10.0f} "
                  f"{a['output_tput_per_gpu']:>9.0f} {a.get('p99_ttft', 0):>12.3f} "
                  f"{a.get('median_tpot', 0):>12.4f}")
    sys.exit(0 if len(ok) == len(summary) else 1)


if __name__ == "__main__":
    main()
