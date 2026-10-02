"""Validate a pinned candidate and the same tests with pinned base production files."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

parser = argparse.ArgumentParser()
parser.add_argument("case")
parser.add_argument("--repo", default="candidate")
args = parser.parse_args()
case = json.loads(Path(args.case).read_text())
repo = Path(args.repo).resolve()
env = dict(os.environ, DEEPFACE_BACKEND_ENGINE="onnx", DEEPFACE_HOME=str(repo / "model-home"))
tests = case["tests"]
results = {}
for variant in ["candidate", "base"]:
    if variant == "base":
        subprocess.run(["git", "checkout", case["base"], "--", *case["production_files"]], cwd=repo, check=True)
    xml = repo / f"{variant}-results.xml"
    run = subprocess.run([sys.executable, "-m", "pytest", *tests, "-q", f"--junitxml={xml}"], cwd=repo, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(run.stdout, flush=True)
    (repo / f"{variant}-tests.log").write_text(run.stdout)
    root = ET.parse(xml).getroot()
    suites = list(root.iter("testsuite"))
    result = {key: sum(int(s.get(key, "0")) for s in suites) for key in ["tests", "failures", "errors", "skipped"]}
    result["exit_code"] = run.returncode
    assert result["tests"] > 0 and result["errors"] == 0 and result["skipped"] == 0, result
    if variant == "candidate":
        assert run.returncode == 0 and result["failures"] == 0, result
    else:
        assert run.returncode == 1 and result["failures"] > 0, result
    results[variant] = result
assert results["base"]["tests"] == results["candidate"]["tests"], results
print(json.dumps({"slug": case["slug"], "head": case["head"], "base": case["base"], "results": results}), flush=True)
