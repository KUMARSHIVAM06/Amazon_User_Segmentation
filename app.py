"""
Amazon User Segmentation — Flask Dashboard (port 5002)
Run:
    python app.py
Then open: http://127.0.0.1:5002
"""
import csv
from pathlib import Path
from flask import Flask, jsonify, render_template, send_from_directory

app = Flask(__name__)

BASE   = Path(__file__).parent
OUT    = BASE / "outputs"
CHARTS = OUT / "charts"


def read_csv(filename):
    path = OUT / filename
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


@app.route("/charts/<path:filename>")
def serve_chart(filename):
    return send_from_directory(CHARTS, filename)


@app.get("/api/summary")
def api_summary():
    segs = read_csv("segment_profile.csv")
    total_users   = sum(int(r["users"]) for r in segs)
    premium       = next((r for r in segs if "Premium" in r["segment"]), {})
    occasional    = next((r for r in segs if "Occasional" in r["segment"]), {})
    deal          = next((r for r in segs if "Deal" in r["segment"]), {})
    return jsonify({
        "total_users":       total_users,
        "total_segments":    len(segs),
        "premium_rev_pct":   premium.get("revenue_pct", 0),
        "premium_user_pct":  premium.get("users_pct", 0),
        "deal_seeker_pct":   deal.get("users_pct", 0),
        "occasional_pct":    occasional.get("users_pct", 0),
        "k_chosen":          5,
    })


@app.get("/api/segments")
def api_segments():
    return jsonify(read_csv("segment_profile.csv"))


@app.get("/api/strategy")
def api_strategy():
    return jsonify(read_csv("segment_strategy.csv"))


@app.get("/api/cross-sell")
def api_cross_sell():
    return jsonify(read_csv("cross_sell_pairs.csv"))


@app.get("/api/k-selection")
def api_k_selection():
    return jsonify(read_csv("k_selection.csv"))


@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True, port=5002)
