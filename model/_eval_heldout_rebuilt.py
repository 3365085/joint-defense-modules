"""诚实对比评测（只读，临时脚本，验证完即删）。

目的：在 dataset_manifest 的 heldout 场景分组留出集上，跑【主项目 rebuilt 内核】
（经真实 VideoDefensePipeline），输出视频级召回/误报，对照 demo 的
"跨场景命中 10/10、干净误报 1/12" 与 "留出误报 8%"。

判据：
  - 攻击视频：在 attack_start+ramp 之后若任一帧 alert_confirmed → 命中(HIT)。
  - 干净视频：整段任一帧 alert_confirmed → 误报(FP)。
  - alert_confirmed 来自 ModuleAResult（3/5 时序确认），即系统实际告警口径。
用法：pixi run python _eval_heldout_rebuilt.py  [--cap 240] [--max-clips N]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_THIS = Path(__file__).resolve()
_SRC = _THIS.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

MANIFEST = _THIS.parents[1] / "rebuilt_demo" / "data" / "dataset_manifest.csv"


def load_heldout() -> list[dict]:
    rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8")))
    return [r for r in rows if r.get("split") == "heldout" and Path(r["path"]).exists()]


def build_pipeline():
    from defense.runtime.pipeline_factory import PipelineCache
    bundle = PipelineCache().get(profile="default")
    return bundle.pipeline


def attack_onset(row: dict) -> int:
    start = int(row.get("attack_start_frame", -1) or -1)
    ramp = int(row.get("attack_ramp_frames", 0) or 0)
    if start < 0:
        return 0
    return start + ramp


def eval_clip(pipe, path: str, cap_frames: int) -> dict:
    pipe.reset()
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return {"ok": False}
    n = 0
    confirmed_frames: list[int] = []
    susp_frames = 0
    while n < cap_frames:
        ok, fr = cap.read()
        if not ok:
            break
        _, _, info = pipe.process_frame(fr)
        if info.get("alert_confirmed"):
            confirmed_frames.append(n)
        if info.get("single_frame_suspicious"):
            susp_frames += 1
        n += 1
    cap.release()
    return {
        "ok": True,
        "frames": n,
        "confirmed_frames": confirmed_frames,
        "n_confirmed": len(confirmed_frames),
        "susp_frames": susp_frames,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=240, help="每段最多处理帧数")
    ap.add_argument("--max-clips", type=int, default=0, help="只跑前 N 段(0=全部)")
    ap.add_argument("--out", default="_eval_heldout_result.json")
    args = ap.parse_args()

    rows = load_heldout()
    if args.max_clips:
        rows = rows[: args.max_clips]
    clean = [r for r in rows if r["label"] == "0"]
    attack = [r for r in rows if r["label"] == "1"]
    print(f"留出集: {len(rows)} 段 = 干净 {len(clean)} + 攻击 {len(attack)}", flush=True)
    print("构建主项目 rebuilt 流水线(default profile)...", flush=True)
    t0 = time.time()
    pipe = build_pipeline()
    print(f"  detector={type(pipe.detector).__name__} impl={pipe.detector_impl} ({time.time()-t0:.1f}s)", flush=True)

    results = []
    fp = 0
    hit = 0
    by_type_hit: dict[str, list[int]] = {}

    print("\n=== 干净视频 (应不报) ===", flush=True)
    for r in clean:
        res = eval_clip(pipe, r["path"], args.cap)
        if not res["ok"]:
            print(f"  SKIP(无法打开) {Path(r['path']).name[:42]}", flush=True)
            continue
        bad = res["n_confirmed"] > 0
        fp += int(bad)
        results.append({**r, **res, "verdict": "FP" if bad else "OK"})
        print(f"  {'FP! ' if bad else 'OK  '} {r['scene_id'][:40]:<42} "
              f"confirmed={res['n_confirmed']}/{res['frames']} susp={res['susp_frames']}", flush=True)

    print("\n=== 跨场景攻击 (应命中, 仅计 onset 之后) ===", flush=True)
    for r in attack:
        res = eval_clip(pipe, r["path"], args.cap)
        if not res["ok"]:
            print(f"  SKIP(无法打开) {Path(r['path']).name[:42]}", flush=True)
            continue
        onset = attack_onset(r)
        post = [f for f in res["confirmed_frames"] if f >= onset]
        ok = len(post) > 0
        hit += int(ok)
        by_type_hit.setdefault(r["attack_type"], []).append(int(ok))
        results.append({**r, **res, "onset": onset, "n_confirmed_post": len(post),
                        "verdict": "HIT" if ok else "MISS"})
        print(f"  {'HIT ' if ok else 'MISS'} {r['attack_type']:<22} {r['scene_id'][:26]:<28} "
              f"onset={onset:>3} confirmed_post={len(post):>3}/{res['frames']}", flush=True)

    n_clean = sum(1 for x in results if x["label"] == "0")
    n_attack = sum(1 for x in results if x["label"] == "1")
    print("\n================ 汇总 ================", flush=True)
    print(f"干净误报 (FP):   {fp}/{n_clean}  = {fp/max(1,n_clean)*100:.1f}%", flush=True)
    print(f"攻击召回 (HIT):  {hit}/{n_attack} = {hit/max(1,n_attack)*100:.1f}%", flush=True)
    print("分攻击类型命中:", flush=True)
    for t, v in sorted(by_type_hit.items()):
        print(f"  {t:<24} {sum(v)}/{len(v)}", flush=True)
    print("\n对照 demo 验收: 跨场景命中 10/10, 干净误报 1/12 (~8%)", flush=True)

    summary = {
        "n_clean": n_clean, "n_attack": n_attack,
        "false_positives": fp, "fp_rate": fp / max(1, n_clean),
        "hits": hit, "recall": hit / max(1, n_attack),
        "by_type_hit": {t: [sum(v), len(v)] for t, v in by_type_hit.items()},
        "cap_frames": args.cap,
        "detector_impl": pipe.detector_impl,
    }
    out = {"summary": summary, "clips": results}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n详细结果 → {args.out}", flush=True)


if __name__ == "__main__":
    main()
