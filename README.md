# Amazon User Segmentation – K-Means Clustering
**Tech:** Python · Pandas · Scikit-learn (K-Means, PCA) · Tableau

8,000 users ke 12 months ke ~1.6 lakh order lines (≈99k orders) se customer behaviour features banaye, K-Means se 5 purchasing segments nikale, har segment ka persona + cross-sell aur retention strategy banayi, aur Tableau dashboard ke liye data export kiya.

> Data note: Amazon ka customer data public nahi hai, isliye `src/generate_data.py` realistic e-commerce data banata hai (Prime, coupons, returns, categories, basket combos). Same pipeline Kaggle ke "Online Retail II" / e-commerce behaviour dataset pe bhi chalegi.

## Run
```bash
pip install -r requirements.txt
python src/generate_data.py     # data/users.csv, data/orders.csv
python src/segmentation.py      # ~10 sec
```

## Approach
1. **Data check** – nulls, duplicates.
2. **Feature engineering (1 row = 1 user):** recency, num_orders, total_spend (returns hata ke), AOV, avg item price, items/order, coupon rate, discount %, return rate, category variety, category shares (item count se), Prime, sessions.
3. **Preprocessing:** skewed columns pe `log1p`, phir `StandardScaler` (K-Means distance-based hai).
4. **K selection:** k = 2–10 pe Elbow (inertia), Silhouette, Davies-Bouldin. Silhouette flat (~0.19–0.21) → elbow bend + business interpretability se **k = 5**.
5. **K-Means** (`n_init=20`) → centroids se rule-based persona naming.
6. **Validation:** PCA 2D plot, profile heatmap.
7. **Cross-sell:** har segment me category pair **lift** = P(A∩B) / (P(A)·P(B)), min 30 orders support.
8. **Strategy + Tableau export.**

## Segments (is run ke)
| Segment | Users % | Revenue % | Pehchaan |
|---|---|---|---|
| Premium Loyalists | 15.4 | 56.3 | ~30 orders/yr, 95% Prime, sabse kam coupon use |
| Tech Enthusiasts | 15.9 | 22.0 | Highest AOV (~₹9.3k), 47% items electronics, 17% return rate |
| Family Bulk Buyers | 15.1 | 11.8 | Frequent, grocery + home heavy, lowest returns |
| Deal Seekers | 29.5 | 6.6 | 84% orders me coupon, low AOV |
| Occasional Shoppers | 24.1 | 3.4 | ~3 orders/yr, 116 din recency, low Prime |

**Key insight:** 15% users (Premium Loyalists) = 56% revenue → inhe retain karna top priority. Occasional + Deal Seekers = 54% users lekin sirf 10% revenue → activation/conversion ka bada mauka.

Top cross-sell pairs: Books + Toys (lift 2–4x), Fashion + Sports, Beauty + Fashion, Grocery + Toys → "Frequently bought together" widgets aur combo offers.

## Outputs
- `outputs/charts/` – elbow/silhouette, PCA scatter, users vs revenue, profile heatmap
- `outputs/segment_profile.csv`, `segment_strategy.csv`, `cross_sell_pairs.csv`, `k_selection.csv`
- `outputs/user_segments_for_tableau.csv`, `orders_with_segment_for_tableau.csv`
- `TABLEAU_GUIDE.md`
