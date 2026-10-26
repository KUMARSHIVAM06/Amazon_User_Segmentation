# Tableau dashboard – steps (Tableau Public free hai)

1. **Connect → Text file** → `outputs/user_segments_for_tableau.csv`. Doosra source add karo: `orders_with_segment_for_tableau.csv` (relationship on `user_id`).
2. **Calculated fields:**
   ```
   Revenue Share   = SUM([Total Spend]) / TOTAL(SUM([Total Spend]))
   User Share      = COUNTD([User Id]) / TOTAL(COUNTD([User Id]))
   AOV             = SUM([Total Spend]) / SUM([Num Orders])
   Is High Value   = IF [Total Spend] > 50000 THEN "High" ELSE "Normal" END
   ```
3. **Sheets:**
   - *Segment Overview* – bar: Segment vs User Share & Revenue Share (dual axis)
   - *KPI tiles* – Users, Revenue, AOV, Prime % per segment (text table)
   - *Behaviour scatter* – X = Num Orders, Y = Total Spend, Color = Segment, Size = AOV
   - *Category mix* – orders source: stacked 100% bar, Segment vs Category (count of line id)
   - *Monthly trend* – orders source: line, MONTH(Order Date) vs SUM(Net Amount), color = Segment
   - *Coupon vs Return* – heatmap: Segment × (Coupon Rate, Return Rate)
   - *Cross-sell table* – `cross_sell_pairs.csv` (Segment, Category A, Category B, Lift)
4. **Dashboard:** sab sheets ek dashboard pe, Segment filter "Apply to all worksheets", Segment bar pe click = filter action.
5. **Story** (optional): 3 points – "Who are they" → "Who pays" → "What to do".
6. Tableau Public pe publish karo, link resume/GitHub pe daalo.
