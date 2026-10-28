"""Regenerate experiments and report artifacts. Run from any directory."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

def run(script, *args):
    log = ROOT / "results" / "logs" / (script.removesuffix(".py") + ("_" + args[-1].split("/")[-1] if args else "") + ".log")
    print(f"Running {script}; log: {log}", flush=True)
    with log.open("w", encoding="utf-8") as stream:
        subprocess.run([sys.executable, str(ROOT / script), *args], cwd=ROOT,
                       env=os.environ.copy(), stdout=stream, stderr=subprocess.STDOUT, check=True)

def validate():
    import pandas as pd
    names = ["de_100", "de_3pixel_100", "de_5pixel_100", "random_100"]
    frames = [pd.read_csv(ROOT / "results" / (n + ".csv")) for n in names]
    ids = set(frames[0].image)
    for name, frame in zip(names, frames):
        if len(frame) != 100 or frame.image.duplicated().any() or set(frame.image) != ids:
            raise RuntimeError(f"{name}: expected exactly the same 100 unique image IDs")
        if not frame.success.isin([True, False]).all():
            raise RuntimeError(f"{name}: invalid success values")
        if not (frame.success == (frame.predicted != frame.true)).all():
            raise RuntimeError(f"{name}: success inconsistent with predictions")
        labels = frames[0].set_index("image").true
        if not frame.set_index("image").true.sort_index().equals(labels.sort_index()):
            raise RuntimeError(f"{name}: inconsistent true labels")
    return sorted(int(i) for i in ids)

def main():
    os.chdir(ROOT)
    os.environ.update(PYTHONHASHSEED="42", CUDA_VISIBLE_DEVICES="-1",
                      TF_DETERMINISTIC_OPS="1", TF_ENABLE_ONEDNN_OPTS="0",
                      MPLBACKEND="Agg", TF_NUM_INTEROP_THREADS="1", TF_NUM_INTRAOP_THREADS="1")
    (ROOT / "results/logs").mkdir(parents=True, exist_ok=True)
    model = ROOT / "networks/models/resnet.h5"
    expected = (ROOT / "model.sha256").read_text().split()[0]
    if not model.exists() or hashlib.sha256(model.read_bytes()).hexdigest() != expected:
        raise RuntimeError("Missing or changed pretrained ResNet; restore networks/models/resnet.h5 from the supplied bundle.")
    outputs = [(1, "de_100"), (3, "de_3pixel_100"), (5, "de_5pixel_100")]
    for pixels, name in outputs:
        run("attack.py", "--model", "resnet", "--pixels", str(pixels),
            "--samples", "100", "--seed", "42", "--maxiter", "75",
            "--popsize", "100", "--save", f"results/{name}.pkl")
    run("random_attack.py", "--samples", "100", "--queries", "7600",
        "--seed", "42", "--save", "results/random_100.pkl")
    ids = validate()
    # Remove only generated figures so a previous run cannot mask missing examples.
    for old in (ROOT / "results/analysis").glob("*.png"):
        old.unlink()
    for old in (ROOT / "results/analysis/examples").glob("*.png"):
        old.unlink()
    run("analyse_results.py")
    run("visualise_results.py")
    analysis = ROOT / "results/analysis"
    required = ["summary_metrics.csv", "pairwise_comparisons.csv", "class_attack_success.csv",
                "attack_success_rates.png", "confidence_reduction.png",
                "examples/de1_best.png", "examples/de1_random_failed.png",
                "examples/de3_best.png", "examples/de5_best.png"]
    for name in required:
        if not (analysis / name).is_file():
            raise RuntimeError(f"Required artifact not generated: {name}. See logs; do not silently substitute an example.")
    manifest = dict(seed=42, samples=100, image_ids=ids, maxiter=75, nominal_popsize=100,
                    actual_populations=[100,90,100], random_candidates_per_image=7600,
                    model_sha256=expected, python=sys.version, platform=platform.platform(),
                    versions={n: importlib.metadata.version(n) for n in
                              ["tensorflow", "keras", "numpy", "pandas", "scipy", "matplotlib"]},
                    source_sha256={n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in
                                   ["attack.py", "random_attack.py", "differential_evolution.py", "helper.py",
                                    "networks/resnet.py", "analyse_results.py", "visualise_results.py", "reproduce.py"]})
    (analysis / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print((analysis / "summary_metrics.csv").read_text())
    print((analysis / "pairwise_comparisons.csv").read_text())
    print("Complete: results/analysis (logs in results/logs).")

if __name__ == "__main__":
    main()
