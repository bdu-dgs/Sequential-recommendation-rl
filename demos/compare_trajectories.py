"""Demo: what different fine-tuned policies actually recommend to the same held-out (test) user.

Reads the greedy test trajectories stored in run metrics (no model loading, no GPU) and writes
reports/demo_trajectories.md: per step, the recommended video, its category, the true simulated reward
(capped FO watch ratio after fatigue) and the logging policy's probability of that item (its "support").
SIMULATED environment (KuaiRecFO-v1).

Usage: python demos/compare_trajectories.py [--seed 0] [--user_index 0]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
POLICIES = [("offline", "cql", "offline CQL (no fine-tuning)"), ("online", "ft_cql", "fine-tune, constant CQL"),
            ("online", "ft_dqn", "fine-tune, no penalty (naive)"),
            ("online", "ft_support_oracle", "fine-tune, support-weighted CQL")]
# KuaiRec first-level category names (Chinese in the raw data) -> English
CATEGORY_EN = {
    "三农": "agriculture", "二次元": "anime", "亲子": "parenting", "人文": "humanities", "健康": "health",
    "健身": "fitness", "其他": "other", "军事": "military", "喜剧": "comedy", "奇人异象": "oddities",
    "娱乐八卦": "gossip", "宠物": "pets", "影视综": "film & TV", "情感": "relationships", "房产家居": "home",
    "摄影": "photography", "教育": "education", "旅游": "travel", "时尚": "fashion", "时政资讯": "politics news",
    "明星娱乐": "celebrities", "星座命理": "astrology", "民生资讯": "local news", "汽车": "cars", "游戏": "games",
    "生活": "lifestyle", "生活泛拍": "daily vlog", "短剧": "short drama", "科学与法律": "science & law",
    "美妆": "beauty", "美食": "food", "自拍": "selfie", "舞蹈": "dance", "艺术": "art", "读书": "books",
    "财经": "finance", "运动": "sports", "音乐": "music", "颜值": "looks", "高新数码": "tech", "UNKNOWN": "unknown"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--user_index", type=int, default=0, help="which stored test trajectory (0-19)")
    ap.add_argument("--steps", type=int, default=12)
    a = ap.parse_args()
    meta = pd.read_parquet(ROOT / "data" / "processed" / "item_meta.parquet")
    cc = pd.read_csv(ROOT / "data" / "raw" / "KuaiRec 2.0" / "data" / "kuairec_caption_category.csv",
                     engine="python", on_bad_lines="skip", usecols=["video_id", "first_level_category_name"])
    cc["video_id"] = pd.to_numeric(cc["video_id"], errors="coerce")
    cat = dict(zip(cc["video_id"], cc["first_level_category_name"]))
    vid = meta["video_id"].values
    out = ["# Demo: four policies, one held-out user (simulated, KuaiRecFO-v1)", "",
           f"Seed {a.seed}, stored test trajectory #{a.user_index}. reward = true simulated reward of the step "
           "(capped real watch ratio × synthetic fatigue); p_log = probability that the logging policy would have "
           "recommended this item in this state. Episode return is the sum of rewards (max 60).", ""]
    for kind, m, label in POLICIES:
        p = ROOT / "experiments" / "runs" / kind / m / f"seed{a.seed}" / "metrics.json"
        if not p.exists():
            continue
        traj = json.loads(p.read_text(encoding="utf-8"))["test"]["trajectories"][a.user_index]
        steps = traj["steps"]
        ret = sum(s[1] for s in steps)
        out += [f"## {label} — return {ret:.1f}, {len(steps)} steps (user row {traj['user_row']})", "",
                "| t | video_id | category | reward | p_log |", "|---|---|---|---|---|"]
        for t, s in enumerate(steps[:a.steps]):
            v = int(vid[s[0]])
            c = cat.get(v, "?")
            out.append(f"| {t} | {v} | {CATEGORY_EN.get(c, c)} | {s[1]:.2f} | {s[3]:.1e} |")
        out.append("")
    dest = ROOT / "reports" / "demo_trajectories.md"
    dest.write_text("\n".join(out) + "\n", encoding="utf-8")
    print("wrote", dest)


if __name__ == "__main__":
    main()
