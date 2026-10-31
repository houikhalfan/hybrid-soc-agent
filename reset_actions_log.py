"""
One-off cleanup: marks every entry in actions_log.json as inactive.

Run this once, with the backend stopped, after confirming via
`sudo iptables -L INPUT -n` that no IPs are actually blocked at the OS
level right now. This resets the dashboard/API's view of "who's blocked"
to match reality (nobody), so you can start fresh once the backend is
running with the privileges it needs to actually apply blocks.

Usage (from your api/ folder, where actions_log.json lives under data/):
    python3 reset_actions_log.py
"""
import json

PATH = "data/actions_log.json"

with open(PATH) as f:
    log = json.load(f)

for entry in log:
    entry["active"] = False

with open(PATH, "w") as f:
    json.dump(log, f, indent=2, default=str)

print(f"Done — {len(log)} entries marked inactive. The dashboard will now show 0 blocked IPs.")
