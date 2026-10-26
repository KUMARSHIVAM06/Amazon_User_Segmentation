"""
Step 2-6: Feature engineering -> scaling -> optimal K (Elbow + Silhouette) -> K-Means ->
          segment profiling/personas -> cross-sell (category lift) -> Tableau exports

Run:  python src/generate_data.py
      python src/segmentation.py
"""
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT, CHARTS = ROOT / "data", ROOT / "outputs", ROOT / "outputs" / "charts"
CHARTS.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")
SNAPSHOT = pd.Timestamp("2026-07-01")
FINAL_K = 5


def log(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


# ------------------------------------------------------------ 1. Load + clean
def load():
    users = pd.read_csv(DATA / "users.csv", parse_dates=["signup_date"])
    orders = pd.read_csv(DATA / "orders.csv", parse_dates=["order_date"])
    log("DATA CHECK")
    print(f"users {users.shape} | order lines {orders.shape}")
    print("nulls:", int(orders.isna().sum().sum()), "| duplicate lines:", int(orders.line_id.duplicated().sum()))
    return users, orders


# ------------------------------------------------------------ 2. Feature engineering (1 row per user)
def build_features(users, orders):
    kept = orders[orders.returned == 0]          # returned items revenue me nahi gine
    g = orders.groupby("user_id")
    f = pd.DataFrame({
        "recency_days": (SNAPSHOT - g.order_date.max()).dt.days,
        "num_orders": g.order_id.nunique(),
        "total_spend": kept.groupby("user_id").net_amount.sum(),
        "avg_item_price": g.mrp.mean(),
        "items_per_order": g.size() / g.order_id.nunique(),
        "coupon_rate": g.coupon_used.mean(),
        "avg_discount_pct": g.discount.sum() / g.mrp.sum(),
        "return_rate": g.returned.mean(),
        "n_categories": g.category.nunique(),
    }).fillna({"total_spend": 0})
    f["aov"] = f.total_spend / f.num_orders
    # Category share by ITEM COUNT (amount se nahi, warna mehenga Electronics har jagah dominate karega)
    share = orders.pivot_table(index="user_id", columns="category", values="line_id",
                               aggfunc="count", fill_value=0)
    share = share.div(share.sum(axis=1), axis=0)
    share.columns = ["share_" + c.lower().replace(" & ", "_").replace(" ", "_") for c in share.columns]
    f = f.join(share).join(users.set_index("user_id")[["is_prime", "monthly_sessions", "wishlist_items",
                                                       "city_tier", "age"]])
    f["conversion_proxy"] = f.num_orders / (f.monthly_sessions * 12)
    return f.reset_index()


CLUSTER_FEATURES = ["recency_days", "num_orders", "total_spend", "avg_item_price", "items_per_order",
                    "coupon_rate", "avg_discount_pct", "return_rate", "n_categories", "is_prime",
                    "monthly_sessions", "share_electronics", "share_grocery", "share_home_kitchen",
                    "share_fashion", "share_toys"]
SKEWED = ["num_orders", "total_spend", "avg_item_price", "recency_days", "monthly_sessions"]


# ------------------------------------------------------------ 3. Choose K
def choose_k(X):
    ks, inertia, sil, dbi = range(2, 11), [], [], []
    sample = np.random.default_rng(0).choice(len(X), min(4000, len(X)), replace=False)
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        inertia.append(km.inertia_)
        sil.append(silhouette_score(X[sample], km.labels_[sample]))
        dbi.append(davies_bouldin_score(X, km.labels_))
    res = pd.DataFrame({"k": ks, "inertia": inertia, "silhouette": sil, "davies_bouldin": dbi}).round(3)
    log("ELBOW + SILHOUETTE"); print(res.to_string(index=False))
    res.to_csv(OUT / "k_selection.csv", index=False)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(res.k, res.inertia, marker="o"); ax[0].set_title("Elbow method (inertia / WCSS)"); ax[0].set_xlabel("k")
    ax[1].plot(res.k, res.silhouette, marker="o", color="#1f8a5b"); ax[1].set_title("Silhouette score"); ax[1].set_xlabel("k")
    fig.tight_layout(); fig.savefig(CHARTS / "01_elbow_silhouette.png", dpi=120); plt.close(fig)
    # Silhouette k=3..10 me lagbhag flat hai (~0.19-0.21), yaani maths akela decide nahi karta.
    # Elbow ka bend k=4-5 pe hai; k=5 pe har cluster ek alag, actionable persona banta hai,
    # jabki k=6-7 pe "Occasional" users 2-3 almost identical groups me toot jaate hain.
    print(f"Chosen k = {FINAL_K} (elbow bend + business interpretability; silhouette is flat)")
    return FINAL_K


# ------------------------------------------------------------ 4. Personas
def name_segments(profile):
    """Cluster centroids dekh kar automatic persona naam (rules relative to overall average)."""
    avg = profile.mean()
    names = {}
    for c, r in profile.iterrows():
        if r.share_electronics > 0.4:
            n = "Tech Enthusiasts"
        elif r.num_orders > 1.6 * avg.num_orders and r.avg_item_price > avg.avg_item_price:
            n = "Premium Loyalists"
        elif r.share_grocery + r.share_home_kitchen > 0.5 and r.num_orders > avg.num_orders:
            n = "Family Bulk Buyers"
        elif r.coupon_rate > 0.65:
            n = "Deal Seekers"
        elif r.num_orders < 0.6 * avg.num_orders:
            n = "Occasional Shoppers"
        else:
            n = "Regular Shoppers"
        while n in names.values():
            n += " II"
        names[c] = n
    return names


STRATEGY = {
    "Tech Enthusiasts": ("Accessories cross-sell (cases, headphones), extended warranty, early-access launches",
                         "Exchange offers on upgrade cycle; reduce returns with better specs/size info"),
    "Premium Loyalists": ("Premium brands, Prime Video/Music bundles, Subscribe & Save on daily items",
                          "VIP perks, priority delivery, loyalty tier – protect this high-value base"),
    "Family Bulk Buyers": ("Bundle packs (grocery + kitchen), toys/baby essentials, Subscribe & Save",
                           "Monthly replenishment reminders, Prime conversion offer for free delivery"),
    "Deal Seekers": ("Lightning deals, combo offers in Fashion/Grocery, 'frequently bought together'",
                     "Push notifications on sales events, coupon wallet; avoid margin-killing blanket discounts"),
    "Occasional Shoppers": ("Personalised recommendations from browsing history, low-price entry categories",
                            "Win-back email/push, first-Prime-month free, festival sale reminders"),
    "Regular Shoppers": ("Category expansion into adjacent categories", "Points / cashback on next order"),
}


# ------------------------------------------------------------ 5. Cross-sell: category lift per segment
def category_lift(orders, seg_map):
    """Lift(A,B) = P(A & B in same order) / (P(A) * P(B)). Lift > 1 => saath me kharidte hain."""
    o = orders.assign(segment=orders.user_id.map(seg_map))
    rows = []
    for seg, d in o.groupby("segment"):
        baskets = d.groupby("order_id").category.apply(set)
        n = len(baskets)
        p = {c: baskets.apply(lambda s: c in s).mean() for c in d.category.unique()}
        for a, b in combinations(sorted(p), 2):
            both = baskets.apply(lambda s: a in s and b in s).mean()
            if both * n >= 30:   # minimum support: kam se kam 30 orders
                rows.append([seg, a, b, round(both, 4), round(both / (p[a] * p[b]), 2)])
    lift = pd.DataFrame(rows, columns=["segment", "category_a", "category_b", "support", "lift"])
    top = lift.sort_values(["segment", "lift"], ascending=[True, False]).groupby("segment").head(3)
    return top


# ------------------------------------------------------------ main
def main():
    users, orders = load()
    feat = build_features(users, orders)

    X = feat[CLUSTER_FEATURES].copy()
    X[SKEWED] = np.log1p(X[SKEWED])                  # skew kam karo (spend heavy-tailed hota hai)
    Xs = StandardScaler().fit_transform(X)           # K-Means distance based hai -> scaling zaroori

    k = choose_k(Xs)
    km = KMeans(n_clusters=k, n_init=20, random_state=42).fit(Xs)
    feat["cluster"] = km.labels_

    profile = feat.groupby("cluster")[CLUSTER_FEATURES + ["aov"]].mean()
    names = name_segments(profile)
    feat["segment"] = feat.cluster.map(names)

    summary = feat.groupby("segment").agg(
        users=("user_id", "count"), avg_orders=("num_orders", "mean"), avg_spend=("total_spend", "mean"),
        aov=("aov", "mean"), coupon_rate=("coupon_rate", "mean"), return_rate=("return_rate", "mean"),
        prime_pct=("is_prime", "mean"), recency=("recency_days", "mean"),
        electronics_share=("share_electronics", "mean"), grocery_share=("share_grocery", "mean"))
    summary["users_pct"] = (100 * summary.users / summary.users.sum()).round(1)
    summary["revenue_pct"] = (100 * feat.groupby("segment").total_spend.sum() / feat.total_spend.sum()).round(1)
    summary = summary.sort_values("avg_spend", ascending=False).round(2)
    log("SEGMENT PROFILE"); print(summary.to_string())
    summary.to_csv(OUT / "segment_profile.csv")

    # Strategy table
    strat = pd.DataFrame([{"segment": s, "users_pct": summary.loc[s, "users_pct"],
                           "revenue_pct": summary.loc[s, "revenue_pct"],
                           "cross_sell": STRATEGY.get(s.replace(" II", ""), ("", ""))[0],
                           "retention": STRATEGY.get(s.replace(" II", ""), ("", ""))[1]} for s in summary.index])
    strat.to_csv(OUT / "segment_strategy.csv", index=False)

    lift = category_lift(orders, dict(zip(feat.user_id, feat.segment)))
    log("TOP CROSS-SELL PAIRS (lift)"); print(lift.to_string(index=False))
    lift.to_csv(OUT / "cross_sell_pairs.csv", index=False)

    # Charts
    pcs = PCA(n_components=2, random_state=0).fit(Xs)
    p2 = pcs.transform(Xs)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=p2[:, 0], y=p2[:, 1], hue=feat.segment, s=8, alpha=0.6, ax=ax, palette="deep")
    ax.set_title(f"Customer segments (PCA 2D view, {pcs.explained_variance_ratio_.sum():.0%} variance)")
    ax.legend(markerscale=3, fontsize=8)
    fig.tight_layout(); fig.savefig(CHARTS / "02_segments_pca.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    summary[["users_pct", "revenue_pct"]].plot(kind="bar", ax=ax, color=["#9db4d8", "#2f6fdb"])
    ax.set_title("Users % vs Revenue % by segment"); ax.tick_params(axis="x", rotation=20)
    fig.tight_layout(); fig.savefig(CHARTS / "03_users_vs_revenue.png", dpi=120); plt.close(fig)

    heat = feat.groupby("segment")[["num_orders", "total_spend", "avg_item_price", "coupon_rate",
                                    "return_rate", "is_prime", "share_electronics", "share_grocery",
                                    "share_fashion", "recency_days"]].mean()
    labels = heat.apply(lambda col: col.map(lambda v: f"{v/1000:.0f}k" if v >= 1000 else
                                             (f"{v:.0f}" if v >= 10 else f"{v:.2f}")))
    fig, ax = plt.subplots(figsize=(12, 4.5))
    sns.heatmap((heat - heat.mean()) / heat.std(), annot=labels, fmt="", cmap="RdBu_r", center=0, ax=ax)
    ax.set_title("Segment profile heatmap (colour = z-score vs other segments, number = actual mean)")
    fig.tight_layout(); fig.savefig(CHARTS / "04_profile_heatmap.png", dpi=120); plt.close(fig)

    # Tableau exports
    feat.to_csv(OUT / "user_segments_for_tableau.csv", index=False)
    orders.merge(feat[["user_id", "segment"]], on="user_id").to_csv(OUT / "orders_with_segment_for_tableau.csv",
                                                                     index=False)
    log("STRATEGY"); print(strat.to_string(index=False))
    print("\nAll outputs saved in outputs/")


if __name__ == "__main__":
    main()
