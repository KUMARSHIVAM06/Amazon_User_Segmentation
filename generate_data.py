"""
Step 1: Amazon jaisa e-commerce customer behaviour data generate karna.
(Real Amazon customer data public nahi hai; same code Kaggle ke 'E-commerce customer behaviour'
 ya 'Online Retail II' dataset pe bhi chalega.)

Output:
  data/users.csv   -> 8,000 users (signup, prime, city tier, sessions, wishlist)
  data/orders.csv  -> order line items (12 months)
"""
from pathlib import Path
import numpy as np
import pandas as pd

RNG = np.random.default_rng(11)
DATA = Path(__file__).resolve().parents[1] / "data"
N_USERS = 8000
START, END = pd.Timestamp("2025-07-01"), pd.Timestamp("2026-06-30")
CATEGORIES = ["Electronics", "Fashion", "Home & Kitchen", "Grocery", "Beauty", "Books", "Sports", "Toys"]
BASE_PRICE = {"Electronics": 6000, "Fashion": 1200, "Home & Kitchen": 1500, "Grocery": 600,
              "Beauty": 700, "Books": 400, "Sports": 1500, "Toys": 900}

COMPLEMENT = {"Grocery": "Home & Kitchen", "Toys": "Grocery", "Fashion": "Beauty",
              "Sports": "Fashion", "Books": "Toys", "Electronics": "Home & Kitchen"}

# Hidden behaviour types (model ko ye label nahi milta, sirf behaviour)
TYPES = {
    #                share, orders/yr, price_mult, discount_lov, return_rate, prime_p, category weights
    "deal_seeker":   (0.28, 10, 0.6, 0.85, 0.10, 0.35, [1, 3, 2, 3, 2, 1, 1, 1]),
    "premium_loyal": (0.15, 30, 1.8, 0.15, 0.05, 0.95, [3, 3, 3, 2, 2, 1, 2, 1]),
    "occasional":    (0.30, 3,  1.0, 0.45, 0.08, 0.15, [2, 2, 2, 1, 1, 2, 1, 1]),
    "tech_buyer":    (0.12, 7,  2.2, 0.35, 0.18, 0.60, [9, 1, 1, 0.5, 0.5, 1, 1, 0.5]),
    "family_bulk":   (0.15, 22, 0.9, 0.55, 0.04, 0.70, [1, 1, 4, 6, 2, 0.5, 0.5, 3]),
}


def main():
    DATA.mkdir(exist_ok=True)
    names = list(TYPES)
    utype = RNG.choice(names, N_USERS, p=[TYPES[t][0] for t in names])
    users = pd.DataFrame({
        "user_id": [f"U{i:05d}" for i in range(1, N_USERS + 1)],
        "signup_date": START - pd.to_timedelta(RNG.integers(30, 2000, N_USERS), unit="D"),
        "city_tier": RNG.choice(["Tier 1", "Tier 2", "Tier 3"], N_USERS, p=[0.45, 0.35, 0.20]),
        "age": RNG.integers(18, 60, N_USERS),
        "is_prime": [int(RNG.random() < TYPES[t][5]) for t in utype],
    })
    users["monthly_sessions"] = [max(1, int(RNG.normal(TYPES[t][1] * 1.5 + 4, 4))) for t in utype]
    users["wishlist_items"] = [int(RNG.poisson(3 + TYPES[t][1] / 3)) for t in utype]

    rows = []
    total_days = (END - START).days
    for uid, t in zip(users.user_id, utype):
        _, freq, pmult, disc, ret, _, w = TYPES[t]
        w = np.array(w, float) * RNG.uniform(0.6, 1.4, len(w))   # har user thoda alag
        n_orders = max(1, RNG.poisson(freq * RNG.uniform(0.6, 1.4)))
        for _ in range(n_orders):
            order_day = START + pd.Timedelta(days=int(RNG.integers(0, total_days)))
            order_id = f"O{len(rows):07d}"
            first = RNG.choice(CATEGORIES, p=w / w.sum())
            basket = [first]
            # Complementary items: real baskets me kuch categories saath aati hain (cross-sell signal)
            if first in COMPLEMENT and RNG.random() < 0.45:
                basket.append(COMPLEMENT[first])
            if RNG.random() < 0.3:
                basket.append(RNG.choice(CATEGORIES, p=w / w.sum()))
            for cat in basket:
                price = BASE_PRICE[cat] * pmult * RNG.lognormal(0, 0.5)
                used_coupon = RNG.random() < disc
                discount = price * RNG.uniform(0.1, 0.45) if used_coupon else price * RNG.uniform(0, 0.05)
                rows.append([order_id, uid, order_day, cat, round(price, 2), round(discount, 2),
                             int(used_coupon), int(RNG.random() < ret)])
    orders = pd.DataFrame(rows, columns=["order_id", "user_id", "order_date", "category", "mrp",
                                         "discount", "coupon_used", "returned"])
    orders["net_amount"] = (orders.mrp - orders.discount).round(2)
    orders.insert(0, "line_id", range(1, len(orders) + 1))
    users.to_csv(DATA / "users.csv", index=False)
    orders.sort_values("order_date").to_csv(DATA / "orders.csv", index=False)
    print(f"users: {len(users):,}  order lines: {len(orders):,}  orders: {orders.order_id.nunique():,}")


if __name__ == "__main__":
    main()
