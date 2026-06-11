# custom-odds

A Jupyter notebook project for analyzing NFL game data and building predictive models for moneyline betting.

## Notebook: odds-comparison.ipynb

### Data Sources
- **nflreadpy** — Python port of the R `nflreadr` package. Returns Polars DataFrames natively.
  - `nflreadpy.load_schedules()` — loads full NFL schedule/results data
  - Installed via `pip3 install nflreadpy`
- **scikit-learn** — logistic regression model
  - Installed via `pip3 install scikit-learn`

### Notebook Structure

**Cell 1 — Imports**
`nflreadpy`, `polars`

**Cell 2 — Season Parameters**
- `season_year`, `week`, `num_weeks`

**Cell 3 — Schedules**
Loads all REG season games via `nflreadpy.load_schedules()`.

**Cell 4 — team_games**
Reshapes the schedules data from one row per game into two rows per game (one per team). Columns: `season`, `week`, `team`, `home_away` (H/A), `points_scored`, `points_allowed`.

**Cell 5 — pred dataframe**
- Filters to seasons 2010–2025, REG games only
- Joins rolling 10-game stats per team (points scored/allowed) for both home and away teams
- Adds `home_diff` and `away_diff` (rolling point differential per team)
- Rolling stats computed with `group_by("team").map_groups()`, sorted ascending by season/week, using `shift(1)` so the current game is excluded

**Cell 6 — Logistic Regression Model**
- Features: `home_diff`, `away_diff`
- Target: `home_win` (1 if home score > away score)
- 80/20 train/test split
- Results: Accuracy ~61%, Log Loss ~0.634, Intercept ~0.22 (captures home field advantage)

**Cell 7 — Predictions and Moneyline Conversion**
Adds 5 columns to `pred`:
- `home_win_prob` — model predicted probability
- `home_moneyline_pred` / `away_moneyline_pred` — moneylines from model probability
- `home_moneyline_no_vig` / `away_moneyline_no_vig` — book moneylines with vig stripped out

Moneyline conversion:
- Favorite (p >= 0.5): `moneyline = -100 * p / (1 - p)`
- Underdog (p < 0.5): `moneyline = 100 * (1 - p) / p`

Vig removal: convert both moneylines to implied probabilities, normalize so they sum to 1.0, convert back to moneylines.

---

## Key Decisions & Notes

- **Polars over pandas** — nflreadpy returns Polars natively, so we stay in Polars throughout. Convert to numpy only at the sklearn boundary.
- **Two features (not one)** — `home_diff` and `away_diff` are kept separate rather than combined into one differential, allowing the model to weight home and away performance independently.
- **Rolling window crosses season boundaries** — the current `team_games` build uses all seasons together, so a team's week 1 stats carry over from the prior season's late weeks. This is a known limitation to revisit.
- **`min_periods` renamed to `min_samples`** in Polars 1.21+.

---

## Modeling Discussion

### Performance Metrics to Track
- **ROI** — simulate flat betting where model disagrees with book by a threshold
- **Calibration** — predicted probability buckets vs actual win rates
- **Brier Score** — mean squared error of probability predictions
- **Expected Value (EV)** — `EV = (p * payout) - (1 - p)` per bet
- **Closing Line Value (CLV)** — does the model beat the closing line? Best evidence of real edge

### Planned Models
1. **Logistic Regression** ✅ — built, using rolling point differential
2. **Elo Ratings** — planned
3. **Linear Regression on spread** — planned

### Ideas for Improving the Model
- Strength of schedule adjustment
- Rest/travel/fatigue features
- Weather and stadium factors
- Injury data (especially QB)
- Predicting against the spread instead of straight win/loss
- Gradient boosting (XGBoost/LightGBM)
- Ensemble of multiple models

### Profitability Notes
- Beating the market requires exceeding the vig (~4-5%) consistently
- Sharp bettors focus on: closing line value, line shopping across books, early reaction to news
- Most public models don't survive transaction costs long-term
