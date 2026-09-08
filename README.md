# Part 2 — Hands-on notebooks

Supervised machine learning for plant pathologists.
Running example: bacterial spot of tomato (*Xanthomonas perforans*), Florida.

## Contents

```
notebooks/
├── data/
│   ├── bacterial_spot_season.csv   480 field-seasons, 8 farms, 10 seasons
│   ├── bacterial_spot_weekly.csv   5,760 plot-weeks
│   └── truth.json                  the generating process, for the Notebook 2 reveal
├── student/                        exercises blanked with #, hand these out
└── instructor/                     the same notebooks, filled in
```

Both versions read from `data/`, and the path resolves whether the notebook is run
from this folder or from `student/` and `instructor/`.

## The three notebooks

| # | Title | Time | What it delivers |
|---|---|---|---|
| 1 | From field data to a feature matrix | ~25 min | What one row means; pipelines; the split exercise |
| 2 | Linear regression, end to end | ~35 min | Reading weights; honest evaluation; regularization; the reveal |
| 3 | Logistic regression and decisions | ~35 min | Odds ratios; decision boundary; thresholds and cost |

Exercises: 4 in Notebook 1, 6 in Notebook 2, 6 in Notebook 3.

## The numbers students should reach

These match the lecture slides exactly, so a student whose output differs has made a
real mistake and can be caught early.

**Notebook 1** — same linear model, three ways of splitting:

| Split | RMSE |
|---|---|
| random 10-fold | 18.50 |
| leave-one-site-out | 18.57 |
| leave-one-year-out | 19.04 |
| predicting the mean | 24.14 |

Random forest on the weekly data: AUC 0.755 random, 0.727 leave-one-plot-out,
0.713 leave-one-year-out.

**Notebook 2** — RMSE 17.99 in-sample, 19.04 leaving out a season, R² 0.44.
Lasso zeroes `soil_ph` and `dist_to_road_m` (both true noise) plus three small real
effects. `max_temp_c` gets a weight of +2.77 despite a true effect of exactly zero.

**Notebook 3** — prevalence 0.143, so "never an event" scores 0.857 accuracy; the
fitted model scores 0.860 and catches 8.7% of events at threshold 0.5.
AUC 0.758. Odds ratios: ×1.70 per 10 h wetness, ×1.64 per 2 °C warmer nights,
×2.10 per log unit of inoculum, ×0.28 for Tygress against FL-8000.

## Setup

Python 3 with `numpy`, `pandas`, `matplotlib` and `scikit-learn` (1.3 or newer — the
notebooks use `sparse_output=` on `OneHotEncoder`). Nothing else is required and
nothing downloads at runtime.

## Two things worth flagging to students

**Notebook 2, section 6.** The reveal is the pedagogical centre of the session. The
model recovers the real coefficients well and still assigns a clear weight to a
variable with no causal role, purely because it correlates 0.93 with a real one. Let
that sit before explaining it.

**Notebook 3, section 10.** The 200:1 cost scenario collapses to "flag almost
everything". That is the correct answer, and it says something useful: for a genuine
zero-tolerance pathogen, a model this uncertain cannot beat blanket action. Knowing
when not to deploy a model is part of the material.
