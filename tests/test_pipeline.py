"""End-to-end smoke test on synthetic data: splits -> train (1 epoch) -> external eval -> acceptance."""
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

from tests.synthetic import make

ROOT = Path(__file__).resolve().parents[1]


def test_end_to_end(tmp_path):
    data = tmp_path / "data"
    make(data)
    splits = tmp_path / "splits"
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    subprocess.run([sys.executable, "-m", "ptx.data.splits", "--data", str(data), "--dev", "candid", "--out", str(splits)],
                   check=True, env=env, cwd=ROOT)
    from ptx.train import load_config, run
    runs = tmp_path / "runs"
    os.environ["PTX_UNLOCK_EXTERNAL"] = "1"
    names = ["full", "abl_no_c1", "abl_no_c2", "abl_no_c3", "abl_no_c4", "abl_base", "bl_unet_r34_kaggle", "bl_groupdro",
             "bl_bandaru_auxcls"]
    for n in names:
        cfg = load_config(ROOT / "configs" / "runs" / f"{n}.yaml")
        cfg["data"].update(root=str(data), splits=str(splits), size=64)
        cfg["model"].update(pretrained=None, encoder="resnet18")
        cfg["train"].update(epochs=1, batch_size=8, workers=0)
        for s in (0, 1):
            run(cfg, s, runs)
    spec = yaml.safe_load(open(ROOT / "configs" / "acceptance.yaml"))
    spec["baselines"] = ["bl_unet_r34_kaggle", "bl_groupdro", "bl_bandaru_auxcls"]
    spec["segmentation_baselines"] = ["bl_unet_r34_kaggle"]
    sp = tmp_path / "acc.yaml"
    yaml.safe_dump(spec, open(sp, "w"))
    out = tmp_path / "results"
    subprocess.run([sys.executable, "-m", "ptx.eval.acceptance", "--spec", str(sp), "--runs", str(runs), "--out", str(out)],
                   check=True, env=env, cwd=ROOT)
    res = json.load(open(out / "acceptance.json"))
    assert "ACCEPTED_AS_POSITIVE" in res["overall"]
    assert (out / "acceptance_report.md").exists()
    assert len(res["criteria"]["A1"]["per_set"]) == 2
