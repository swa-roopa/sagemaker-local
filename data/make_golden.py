"""Generate a synthetic stand-in for the Day 1 golden dataset.

Shape matches the Day 3 plan: 265 rows (one every 5 s), 157 normal rows,
36 rows each of latency / errors / cpu incidents, and 108 `transition` rows.

Metrics are 1-minute rolling averages (12 samples) of the raw signal, while the
label switches instantly. That lag is what makes the transition rows ambiguous:
the first 30 s of an incident still look normal, and the minute after it ends
still looks like an incident.
"""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
WINDOW = 12          # 12 x 5 s = 1-minute rate window
INCIDENT_LEN = 18    # 90 s per incident
GAP = 24             # 2 min of normal after each incident
LEAD = 13            # normal rows before the first incident
CYCLE = ["latency", "errors", "cpu"] * 2

labels = ["normal"] * LEAD
for kind in CYCLE:
    labels += [kind] * INCIDENT_LEN + ["normal"] * GAP

raw = []
for kind in labels:
    req, err, p95, cpu = 15.0, 0.002, 0.08, 0.10
    if kind == "latency":
        req, p95 = 6.0, 1.4
    elif kind == "errors":
        err = 0.30
    elif kind == "cpu":
        cpu = 0.90
    raw.append([
        req * rng.normal(1, 0.06),
        max(0.0, err + rng.normal(0, 0.002)),
        p95 * rng.normal(1, 0.08),
        min(1.0, max(0.0, cpu + rng.normal(0, 0.02))),
    ])

df = pd.DataFrame(raw, columns=["req_rate", "err_rate", "p95_latency", "cpu"])
df = df.rolling(WINDOW, min_periods=1).mean().round(4)
df.insert(0, "ts", pd.date_range("2026-10-26 15:00:00", periods=len(df), freq="5s").astype(str))
df["incident_type"] = labels
df["is_incident"] = (df["incident_type"] != "normal").astype(int)

# transition = first 30 s of each incident + the minute after it ends
transition = np.zeros(len(df), dtype=bool)
start = LEAD
for _ in CYCLE:
    transition[start:start + 6] = True
    transition[start + INCIDENT_LEN:start + INCIDENT_LEN + WINDOW] = True
    start += INCIDENT_LEN + GAP
df["transition"] = transition

df = df[["ts", "req_rate", "err_rate", "p95_latency", "cpu", "is_incident", "incident_type", "transition"]]
out = Path(__file__).parent / "golden" / "incident_dataset.csv"
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
print(f"wrote {out} shape={df.shape}")
print(df["incident_type"].value_counts().to_dict(), "transition rows:", int(df["transition"].sum()))
