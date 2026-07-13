from __future__ import annotations
import os, sys
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from defense.module_a.backends import create_detector_backend
from defense.runtime.config import PROJECT_ROOT, load_runtime_config
from defense.runtime.a3b_soft_trigger import A3BSoftTriggerState
from defense.pipelines.video_defense_pipeline import VideoDefensePipeline


def main():
    config = load_runtime_config()
    backend = create_detector_backend(config, PROJECT_ROOT)
    pipeline = VideoDefensePipeline(backend, config=config)
    pipeline.warmup()
    a3b_state = A3BSoftTriggerState(config.get("a3b", {}))

    video_path = "D:/联合防御模块/素材/视频中出现干扰视频/VID20260512200916.mp4"
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0

    fi = 0
    fa3b = fsusp = falert = None
    p_media_high = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        _, _, info = pipeline.process_frame(frame)
        det = info.get("details", {})
        feats = det.get("module_a_features", {})
        si = dict(feats.get("static_media") or feats.get("static_image", {}))
        si["source_path"] = str(video_path)
        r = a3b_state.update(si)
        trig = bool(r.get("triggered", False))
        susp = info.get("is_attack", False)  # to_info_dict uses is_attack
        alert = info.get("alert_confirmed", False)
        obs = float(r.get("observed_score", 0.0))
        p_media = float(si.get("p_media", 0.0))

        if trig and fa3b is None: fa3b = fi
        if susp and fsusp is None: fsusp = fi
        if alert and falert is None: falert = fi

        if p_media >= 0.40:
            p_media_high.append((fi, p_media, trig, alert))

        # Print timeline every 2 seconds
        if fi % 120 == 0:
            print(f"  {fi:5d} ({fi/fps:.1f}s)  p_media={p_media:.3f}  trig={trig}  alert={alert}  obs={obs:.3f}")

        fi += 1
        if fi >= 2618:
            break

    print(f"\nAll A3b p_media>=0.40 events:")
    for f, s, t, a in p_media_high[:30]:
        print(f"  frame {f:5d} ({f/fps:.1f}s)  p_media={s:.3f}  trig={t}  alert={a}")
    if len(p_media_high) > 30:
        print(f"  ... and {len(p_media_high)-30} more events")

    cap.release()
    pipeline.close()

    if falert:
        print(f"\nFirst alert: frame {falert} = {falert/fps:.2f}s")
    print(f"First A3b triggered: frame {fa3b} = {fa3b/fps:.2f}s" if fa3b else "No A3b triggered")
    print(f"Total frames with p_media>=0.40: {len(p_media_high)}")


if __name__ == "__main__":
    main()
