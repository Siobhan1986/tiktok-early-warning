# main.py
"""
Storyful Intelligence — TikTok Early Warning (MVP prototype, search-enabled)
----------------------------------------------------------------------------
Adds a 'search' mode so a user can query a brand and receive:
- Narratives clustered (sound/hashtag) ranked by Narrative Risk Score (NRS)
- Within each narrative, top influencers ranked by Influence Score (IS)

Usage:
  python main.py           # continuous engine mode (alerts)
  python main.py search    # interactive brand search (ranked clusters)

Replace `fetch_videos_from_ensemble()` with your real Ensemble API call.
"""

from __future__ import annotations
import sys, time, random, math, json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import defaultdict, deque
from datetime import datetime, timedelta

# ---------- Config ----------

MARKET_CONFIG = {
    "US": {"nrs_threshold": 65, "zscore_threshold": 2.0, "min_unique_creators": 5, "window_minutes": 15, "baseline_minutes": 120},
    "UK": {"nrs_threshold": 60, "zscore_threshold": 2.0, "min_unique_creators": 4, "window_minutes": 15, "baseline_minutes": 120},
}

WATCHLISTS = {
    "US": {"brands": ["storyful", "brandx", "brand y"], "hashtags": ["#brandx", "#recall", "#boycottbrandx"],
           "sounds": ["snd-eco-123", "snd-brandx-999"], "creators": ["influencer_a", "watchdog_news"], "effects": ["fx_glitch_42"]},
    "UK": {"brands": ["storyful", "brandz"], "hashtags": ["#brandz", "#shortage"], "sounds": [], "creators": [], "effects": []},
}

# ---------- Data Models ----------

@dataclass
class Video:
    id: str
    caption: str
    creator_id: str
    creator_followers: int
    views: int
    likes: int
    shares: int
    comments: int
    sound_id: Optional[str]
    effect_id: Optional[str]
    hashtags: List[str]
    market: str
    ts: datetime

@dataclass
class CreatorStats:
    views: int = 0
    likes: int = 0
    shares: int = 0
    comments: int = 0
    posts: int = 0
    followers: int = 0

@dataclass
class Narrative:
    id: str            # "sound:<id>" or "hashtag:<#tag>"
    kind: str          # "sound" | "hashtag"
    market: str
    videos: List[Video] = field(default_factory=list)

# ---------- Simulated Ensemble Data API ----------

import os, requests
from datetime import datetime, timezone

ENSEMBLE_BASE_URL = os.getenv("ENSEMBLE_BASE_URL", "https://api.ensembledata.com")
ENSEMBLE_API_KEY = os.getenv("ENSEMBLE_API_KEY")

def fetch_videos_from_ensemble(market: str, since: datetime, until: datetime) -> List[Video]:
    """
    Real Ensemble Data API call.
    Adjust endpoint/params/field names to match your tenant’s spec.
    Supports simple pagination via 'next' cursor if provided.
    """
    if ENSEMBLE_API_KEY is None:
        raise RuntimeError("Missing ENSEMBLE_API_KEY secret.")

    headers = {
        "Authorization": f"Bearer {ENSEMBLE_API_KEY}",
        "Accept": "application/json",
    }

    # Example endpoint/params — tweak to match your API:
    url = f"{ENSEMBLE_BASE_URL}/tiktok/videos"
    params = {
        "market": market,
        "since": since.replace(tzinfo=timezone.utc).isoformat().replace("+00:00","Z"),
        "until": until.replace(tzinfo=timezone.utc).isoformat().replace("+00:00","Z"),
        "limit": 200,  # adjust per API limits
        # Optional: filters for hashtags/sounds/keywords if supported
        # "q": "brandx OR brandy",
    }

    out: List[Video] = []
    next_cursor = None

    while True:
        qp = dict(params)
        if next_cursor:
            qp["cursor"] = next_cursor

        r = requests.get(url, headers=headers, params=qp, timeout=20)
        r.raise_for_status()
        data = r.json()

        rows = data.get("videos", data.get("data", []))  # adapt to your response shape
        for v in rows:
            # Map API fields → our Video model (rename as needed)
            out.append(Video(
                id=str(v.get("id")),
                caption=v.get("caption", "") or "",
                creator_id=str(v.get("creator", {}).get("id") or v.get("author_id") or "unknown"),
                creator_followers=int(v.get("creator", {}).get("followers", v.get("followers", 0)) or 0),
                views=int(v.get("metrics", {}).get("views", v.get("views", 0)) or 0),
                likes=int(v.get("metrics", {}).get("likes", v.get("likes", 0)) or 0),
                shares=int(v.get("metrics", {}).get("shares", v.get("shares", 0)) or 0),
                comments=int(v.get("metrics", {}).get("comments", v.get("comments", 0)) or 0),
                sound_id=v.get("sound", {}).get("id") or v.get("sound_id"),
                effect_id=v.get("effect", {}).get("id") or v.get("effect_id"),
                hashtags=[("#" + h.lstrip("#")).lower() for h in (v.get("hashtags") or [])],
                market=market,
                ts=datetime.fromisoformat((v.get("created_at") or v.get("timestamp")).replace("Z","+00:00")),
            ))

        next_cursor = data.get("next") or data.get("next_cursor")
        if not next_cursor:
            break

    return out


# ---------- Narrative Builder ----------

def top_hashtag(hashtags: List[str]) -> Optional[str]:
    return hashtags[0] if hashtags else None

def build_narratives(videos: List[Video], market: str) -> Dict[str, Narrative]:
    narratives: Dict[str, Narrative] = {}
    for v in videos:
        if v.market != market: continue
        if v.sound_id:
            key, kind = f"sound:{v.sound_id}", "sound"
        else:
            th = top_hashtag(v.hashtags)
            if not th: continue
            key, kind = f"hashtag:{th.lower()}", "hashtag"
        narratives.setdefault(key, Narrative(id=key, kind=kind, market=market)).videos.append(v)
    return narratives

# ---------- Scoring ----------

NEGATIVE_LEXICON = {"boycott","avoid","concern","warning","scandal","fraud","recall","ban"}
POSITIVE_LEXICON = {"support","love","great","amazing","win","thanks"}

def estimate_stance(text: str) -> int:
    t = text.lower(); s = 0
    for w in NEGATIVE_LEXICON:
        if w in t: s -= 1
    for w in POSITIVE_LEXICON:
        if w in t: s += 1
    return 1 if s > 0 else (-1 if s < 0 else 0)

def brand_proximity(text: str, market: str, brand_query: Optional[str]=None) -> float:
    t = text.lower()
    wl = WATCHLISTS.get(market, {})
    brands = [b.lower() for b in wl.get("brands", [])]
    if brand_query: brands.append(brand_query.lower())
    if any(b in t for b in brands): return 1.0
    return 0.0

def compute_is_per_creator(videos: List[Video]) -> Dict[str, float]:
    per: Dict[str, CreatorStats] = defaultdict(CreatorStats)
    for v in videos:
        s = per[v.creator_id]
        s.views += v.views; s.likes += v.likes; s.shares += v.shares; s.comments += v.comments; s.posts += 1
        s.followers = max(s.followers, v.creator_followers)
    out: Dict[str, float] = {}
    for cid, s in per.items():
        engagement = s.likes + 2*s.shares + 1.5*s.comments
        resonance = engagement / max(1, s.views)
        reach = math.log10(max(10, s.followers)) / 6.0
        momentum = min(1.0, s.posts / 10.0)
        raw = 0.5*resonance + 0.3*reach + 0.2*momentum
        out[cid] = max(0.0, min(100.0, raw*100))
    return out

def compute_nrs(narr: Narrative, market: str, z: float, brand_query: Optional[str]=None) -> Tuple[float, int, int]:
    vids = narr.videos
    velocity = len(vids)
    reach = sum(v.views for v in vids)
    creators = {v.creator_id for v in vids}
    stance_vals = [estimate_stance(v.caption) for v in vids]
    stance_score = (sum(stance_vals) / max(1, len(stance_vals)) + 1) / 2  # 0..1 (1 = positive)
    brand_prox = max(brand_proximity(v.caption, market, brand_query) for v in vids) if vids else 0.0
    source_cred = min(1.0, len(creators)/10.0)
    vel_n = min(1.0, velocity/20.0)
    reach_n = min(1.0, reach/500_000.0)
    z_n = min(1.0, max(0.0, z/5.0))
    nrs = (0.35*max(vel_n,z_n) + 0.20*reach_n + 0.15*source_cred + 0.15*brand_prox + 0.10*(1 - stance_score) + 0.05*z_n)*100.0
    return nrs, velocity, reach

# ---------- Rolling stats (for engine mode only) ----------

class RollingStats:
    def __init__(self, baseline_minutes: int, window_minutes: int):
        self.baseline = deque()
        self.window = deque()
        self.baseline_minutes = baseline_minutes
        self.window_minutes = window_minutes

    def add(self, ts: datetime, count: int):
        self.baseline.append((ts, count)); self.window.append((ts, count)); self._trim()

    def _trim(self):
        now = datetime.utcnow()
        while self.baseline and (now - self.baseline[0][0]).total_seconds() > self.baseline_minutes*60:
            self.baseline.popleft()
        while self.window and (now - self.window[0][0]).total_seconds() > self.window_minutes*60:
            self.window.popleft()

    def zscore(self) -> float:
        vals = [c for _, c in self.baseline]
        if len(vals) < 5: return 0.0
        mean = sum(vals)/len(vals)
        var = sum((x-mean)**2 for x in vals)/len(vals)
        std = math.sqrt(var) or 1.0
        recent = sum(c for _, c in self.window)
        return (recent - mean)/std

# ---------- Alerting (engine mode) ----------

def make_alert(narr: Narrative, market: str, nrs: float, z: float, velocity: int, reach: int, iscores: Dict[str, float]) -> Dict:
    lineage = {}
    if narr.kind == "sound": lineage["sound_id"] = narr.id.split(":",1)[1]
    else: lineage["hashtag"] = narr.id.split(":",1)[1]
    effects = list({v.effect_id for v in narr.videos if v.effect_id})[:5]
    if effects: lineage["effects"] = effects

    top_drivers = sorted(iscores.items(), key=lambda x: x[1], reverse=True)[:5]
    drivers = []
    for cid, score in top_drivers:
        vids = [v for v in narr.videos if v.creator_id == cid]
        stance_vals = [estimate_stance(v.caption) for v in vids]
        stance = "negative" if sum(stance_vals) < 0 else ("positive" if sum(stance_vals) > 0 else "mixed/neutral")
        drivers.append({"creator_id": cid, "influence_score": round(score,2), "stance": stance})

    stance_vals_all = [estimate_stance(v.caption) for v in narr.videos]
    overall_stance = "negative" if sum(stance_vals_all) < 0 else ("positive" if sum(stance_vals_all) > 0 else "mixed/neutral")
    completeness = min(1.0, len(narr.videos)/20.0)
    confidence = round(50 + 50*completeness, 1)

    return {
        "narrative_id": narr.id,
        "market": market,
        "summary": f"{narr.kind}-led narrative trending in {market}",
        "trajectory": {"posts_window": velocity, "views_window": reach, "zscore_burst": round(z,2)},
        "lineage": lineage,
        "top_drivers": drivers,
        "stance": overall_stance,
        "confidence": confidence,
        "recommended_next_steps": ["Prepare holding lines; brief exec sponsor.","Engage neutral creators with context.","Publish verified clarification if warranted."],
        "nrs": round(nrs,1),
        "timestamp": datetime.utcnow().isoformat()+"Z"
    }

def emit_alert(payload: Dict):
    print("\n=== PRIORITY ALERT ===")
    print(json.dumps(payload, indent=2))

# ---------- Engine Mode ----------

class EarlyWarningEngine:
    def __init__(self, market: str):
        self.market = market
        cfg = MARKET_CONFIG[market]
        self.stats_by_narr: Dict[str, RollingStats] = {}
        self.window_minutes = cfg["window_minutes"]
        self.baseline_minutes = cfg["baseline_minutes"]
        self.nrs_threshold = cfg["nrs_threshold"]
        self.z_threshold = cfg["zscore_threshold"]
        self.min_unique_creators = cfg["min_unique_creators"]

    def process_batch(self, videos: List[Video]):
        narratives = build_narratives(videos, self.market)
        for nid, narr in narratives.items():
            rs = self.stats_by_narr.get(nid) or RollingStats(self.baseline_minutes, self.window_minutes)
            self.stats_by_narr[nid] = rs
            rs.add(datetime.utcnow(), len(narr.videos))
            z = rs.zscore()
            iscores = compute_is_per_creator(narr.videos)
            nrs, velocity, reach = compute_nrs(narr, self.market, z)
            unique_creators = len({v.creator_id for v in narr.videos})
            trending = (z >= self.z_threshold and unique_creators >= self.min_unique_creators) or (nrs >= self.nrs_threshold)
            if trending:
                payload = make_alert(narr, self.market, nrs, z, velocity, reach, iscores)
                emit_alert(payload)

def run_engine(market: str = "US", tick_seconds: int = 5):
    eng = EarlyWarningEngine(market)
    print(f"[engine] Starting market={market} (tick={tick_seconds}s). Ctrl+C to stop.")
    try:
        while True:
            now = datetime.utcnow()
            since = now - timedelta(seconds=tick_seconds)
            batch = fetch_videos_from_ensemble(market, since, now)
            eng.process_batch(batch)
            time.sleep(tick_seconds)
    except KeyboardInterrupt:
        print("\n[engine] Stopped.")

# ---------- Search Mode (NEW) ----------

def run_search_interactive():
    print("=== TikTok Narrative Search (prototype) ===")
    market = input(f"Market ({'/'.join(MARKET_CONFIG.keys())}) [US]: ").strip() or "US"
    brand = input("Brand/Entity to search (e.g., brandx): ").strip()
    lookback_min = int(input("Lookback minutes [120]: ").strip() or "120")
    top_k = int(input("How many narratives to show [5]: ").strip() or "5")
    if not brand:
        print("Please enter a brand to search."); return

    until = datetime.utcnow()
    since = until - timedelta(minutes=lookback_min)

    # Replace with real Ensemble API call
    videos = fetch_videos_from_ensemble(market, since, until)

    # Basic brand filter: keep videos where caption/hashtags mention the brand
    brand_l = brand.lower().lstrip("#")
    filtered = []
    for v in videos:
        in_caption = brand_l in v.caption.lower()
        in_tags = any(brand_l in h.lower().lstrip("#") for h in v.hashtags)
        if in_caption or in_tags:
            filtered.append(v)

    if not filtered:
        print("No videos matched your brand in the selected window."); return

    narratives = build_narratives(filtered, market)

    # Compute pseudo-burst z for ranking within this lookback (single-window approximation)
    is_by_narr: Dict[str, Dict[str,float]] = {}
    scored: List[Tuple[str, float, int, int]] = []
    for nid, narr in narratives.items():
        iscores = compute_is_per_creator(narr.videos)
        is_by_narr[nid] = iscores
        # single-window z approx: use narrative size vs average across narratives
        # compute mean posts across narratives to get a rough z
        # (not a true baseline; good enough for ranked search)
    mean_posts = max(1.0, sum(len(n.videos) for n in narratives.values()) / max(1, len(narratives)))
    for nid, narr in narratives.items():
        z_approx = (len(narr.videos) - mean_posts) / max(1.0, math.sqrt(mean_posts))
        nrs, velocity, reach = compute_nrs(narr, market, z_approx, brand_query=brand)
        scored.append((nid, nrs, velocity, reach))

    ranked = sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]

    # Present results
    print(f"\n=== Top Narratives for '{brand}' in {market} (last {lookback_min}m) ===")
    for idx, (nid, nrs, velocity, reach) in enumerate(ranked, start=1):
        narr = narratives[nid]
        lineage = ("sound: " + narr.id.split(":",1)[1]) if narr.kind == "sound" else ("hashtag: " + narr.id.split(":",1)[1])
        stance_vals = [estimate_stance(v.caption) for v in narr.videos]
        stance_label = "negative" if sum(stance_vals) < 0 else ("positive" if sum(stance_vals) > 0 else "mixed/neutral")
        print(f"\n[{idx}] {lineage}")
        print(f"    NRS: {nrs:.1f} | posts: {len(narr.videos)} | views: {reach} | stance: {stance_label}")
        # top influencers by IS
        drivers = sorted(is_by_narr[nid].items(), key=lambda x: x[1], reverse=True)[:5]
        print("    Top influencers (by IS):")
        for cid, sc in drivers:
            print(f"      - {cid:16s}  IS={sc:5.1f}")

    print("\nTip: replace fetch_videos_from_ensemble() with your Ensemble API to get real results.")

# ---------- Entry ----------

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].lower() == "search":
        run_search_interactive()
    else:
        run_engine(market="US", tick_seconds=5)
