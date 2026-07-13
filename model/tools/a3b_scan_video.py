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
    fa3b = falert = fsusp = None
    p_media_events = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        _, _, info = pipeline.process_frame(frame)
        feats = info.get("details", {}).get("module_a_features", {})
        si = dict(feats.get("static_media") or feats.get("static_image", {}))
        si["source_path"] = video_path
        r = a3b_state.update(si)

        trig = bool(r.get("triggered", False))
        alert = info.get("alert_confirmed", False)
        susp = info.get("is_attack", False)  # to_info_dict uses is_attack
        obs = float(r.get("observed_score", 0.0))
        p_media = float(si.get("p_media", 0.0))

        if trig and fa3b is None: fa3b = fi
        if alert and falert is None: falert = fi
        if susp and fsusp is None: fsusp = fi

        if p_media >= 0.40:
            p_media_events.append((fi, p_media, trig, alert))

        # Timeline every 5s
        if fi % 300 == 0:
            print(f"  {fi:5d} ({(fi/fps):.0f}s)  p_media={p_media:.3f}  trig={trig}  alert={alert}")

        fi += 1
        if fi >= 2618:
            break

    cap.release()
    pipeline.close()

    print(f"\n=== Full video scan: {fi} frames at {fps:.0f}fps ===")
    print(f"First A3BSoftTrigger triggered: frame {fa3b} = {fa3b/fps:.2f}s" if fa3b else "No A3BSoftTrigger triggered EVER")
    print(f"First suspicious: frame {fsusp} = {fsusp/fps:.2f}s" if fsusp else "No suspicious EVER")
    print(f"First alert: frame {falert} = {falert/fps:.2f}s" if falert else "No alert EVER")
    print(f"Frames with p_media>=0.40: {len(p_media_events)}")

    if p_media_events:
        # Show clusters
        print(f"\np_media>=0.40 clusters:")
        cluster_start = p_media_events[0][0]
        for i, (fi, pm, tr, al) in enumerate(p_media_events):
            if i > 0 and fi - p_media_events[i-1][0] > 30:
                print(f"  cluster: {cluster_start} ({cluster_start/fps:.1f}s) to {p_media_events[i-1][0]} ({p_media_events[i-1][0]/fps:.1f}s)")
                cluster_start = fi
        # last cluster
        print(f"  cluster: {cluster_start} ({cluster_start/fps:.1f}s) to {p_media_events[-1][0]} ({p_media_events[-1][0]/fps:.1f}s)")

if __name__ == "__main__":
    main()
