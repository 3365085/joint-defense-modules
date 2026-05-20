(function (root, factory) {
  if (typeof module === "object" && module.exports) {
    module.exports = factory();
  } else {
    const api = factory();
    root.OverlayTimeline = api.OverlayTimeline;
    root.interpolateOverlay = api.interpolateOverlay;
  }
}(typeof globalThis !== "undefined" ? globalThis : window, function () {
  function number(value, fallback) {
    const n = Number(value);
    return Number.isFinite(n) ? n : fallback;
  }

  function cloneTrack(track) {
    return Object.assign({}, track || {}, { box: Array.from((track && track.box) || []) });
  }

  function lerp(a, b, t) {
    return number(a, 0) + (number(b, 0) - number(a, 0)) * t;
  }

  function boxLerp(a, b, t) {
    const out = [];
    for (let i = 0; i < 4; i += 1) out.push(lerp(a && a[i], b && b[i], t));
    return out;
  }

  function interpolateOverlay(prev, next, t) {
    const leftTime = number(prev && prev.video_time_s, NaN);
    const rightTime = number(next && next.video_time_s, NaN);
    if (!Number.isFinite(leftTime) || !Number.isFinite(rightTime) || rightTime <= leftTime) {
      return null;
    }
    const ratio = Math.max(0, Math.min(1, (number(t, leftTime) - leftTime) / (rightTime - leftTime)));
    const nextById = new Map(((next && next.ppe_tracks) || []).map((track) => [String(track.track_id ?? track.id ?? ""), track]));
    const tracks = ((prev && prev.ppe_tracks) || []).map((track) => {
      const key = String(track.track_id ?? track.id ?? "");
      const later = nextById.get(key);
      if (!later || !Array.isArray(track.box) || !Array.isArray(later.box)) return cloneTrack(track);
      const mixed = cloneTrack(track);
      mixed.box = boxLerp(track.box, later.box, ratio);
      mixed.confidence = lerp(track.confidence, later.confidence, ratio);
      mixed.misses = Math.round(lerp(track.misses, later.misses, ratio));
      mixed.source = mixed.source || "tracked";
      return mixed;
    });
    return Object.assign({}, prev, {
      video_time_s: number(t, leftTime),
      ppe_tracks: tracks,
      interpolated: true,
    });
  }

  class OverlayTimeline {
    constructor(options) {
      const opts = options || {};
      this.items = [];
      this.maxItems = number(opts.maxItems, 600);
      this.lastHeld = null;
    }

    clear() {
      this.items = [];
      this.lastHeld = null;
    }

    push(record) {
      if (!record || !Number.isFinite(number(record.video_time_s, NaN))) return;
      const seq = number(record.overlay_seq ?? record.seq, 0);
      const videoTime = number(record.video_time_s, 0);
      const exists = this.items.some((item) => {
        const itemSeq = number(item.overlay_seq ?? item.seq, -1);
        return itemSeq === seq && Math.abs(number(item.video_time_s, -9999) - videoTime) < 0.001;
      });
      if (exists) return;
      this.items.push(record);
      this.items.sort((a, b) => number(a.video_time_s, 0) - number(b.video_time_s, 0));
      if (this.items.length > this.maxItems) {
        this.items.splice(0, this.items.length - this.maxItems);
      }
    }

    findNearest(t, windowSec) {
      const time = number(t, NaN);
      if (!Number.isFinite(time)) return null;
      let best = null;
      let bestDt = Infinity;
      for (const item of this.items) {
        const dt = Math.abs(number(item.video_time_s, NaN) - time);
        if (dt < bestDt) {
          best = item;
          bestDt = dt;
        }
      }
      if (!best || bestDt > windowSec) return null;
      this.lastHeld = best;
      return best;
    }

    findBracket(t, maxGapSec) {
      const time = number(t, NaN);
      if (!Number.isFinite(time)) return null;
      let prev = null;
      let next = null;
      for (const item of this.items) {
        const itemTime = number(item.video_time_s, NaN);
        if (!Number.isFinite(itemTime)) continue;
        if (itemTime <= time) prev = item;
        if (itemTime > time) {
          next = item;
          break;
        }
      }
      if (!prev || !next) return null;
      if ((number(next.video_time_s, 0) - number(prev.video_time_s, 0)) > maxGapSec) return null;
      return { prev, next };
    }

    heldOverlayIfFresh(t, holdSec) {
      const time = number(t, NaN);
      if (!this.lastHeld || !Number.isFinite(time)) return null;
      const dt = time - number(this.lastHeld.video_time_s, NaN);
      if (dt < 0 || dt > holdSec) return null;
      return Object.assign({}, this.lastHeld, {
        video_time_s: time,
        held: true,
        ppe_tracks: ((this.lastHeld.ppe_tracks) || []).map((track) => {
          const copy = cloneTrack(track);
          if (!copy.source || copy.source === "detected") copy.source = "held";
          return copy;
        }),
      });
    }

    select(t, options) {
      const opts = options || {};
      const matchWindowSec = number(opts.matchWindowSec, 0.18);
      const interpolateSec = number(opts.interpolateSec, 0.4);
      const holdSec = number(opts.holdSec, 0.55);
      const maxAgeSec = number(opts.maxAgeSec, 0.95);
      // Prefer interpolation while the display clock is between two detector
      // records. Nearest-first selection produces visible step/drag at 10~15 FPS.
      const bracket = this.findBracket(t, interpolateSec);
      if (bracket) {
        const mixed = interpolateOverlay(bracket.prev, bracket.next, t);
        if (mixed) {
          this.lastHeld = mixed;
          return mixed;
        }
      }
      const nearest = this.findNearest(t, matchWindowSec);
      if (nearest) return nearest;
      const held = this.heldOverlayIfFresh(t, Math.min(holdSec, maxAgeSec));
      if (held) return held;
      return null;
    }
  }

  return { OverlayTimeline, interpolateOverlay };
}));
