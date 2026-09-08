"""
Synthetic data generator: bacterial spot of tomato (Xanthomonas perforans), Florida.

Ground truth is fully documented in truth.json so the lecture can reveal, at the end
of the notebook session, exactly what the model should have recovered.

Design choices are pedagogical, not arbitrary:
  * site and year random effects  -> spatial/temporal autocorrelation, so a random
    train/test split leaks and a leave-one-year-out split does not.
  * mean_temp_c and max_temp_c correlated ~0.93 -> collinearity, sign flips in the
    coefficient table (Discussion 2).
  * soil_ph and dist_to_road_m are pure noise -> lasso should zero them out.
  * severity squashed through a logistic ceiling -> the 100% ceiling effect that real
    severity ratings show (observed-vs-predicted diagnostic slide).
"""

import json
import numpy as np
import pandas as pd

RNG = np.random.default_rng(20260908)

# ----------------------------------------------------------------------------- design
YEARS = list(range(2016, 2026))          # 10 seasons
SITES = [f"S{i:02d}" for i in range(1, 9)]  # 8 farms
COUNTY = {"S01": "Manatee", "S02": "Manatee", "S03": "Hillsborough", "S04": "Hillsborough",
          "S05": "Collier", "S06": "Collier", "S07": "Miami-Dade", "S08": "Miami-Dade"}
PLOTS_PER_SITE_YEAR = 6

CULTIVARS = ["FL-8000", "Sanibel", "Tygress", "HM-1823", "Sebring"]
# true additive cultivar effect on the latent severity scale (partial resistance)
CULTIVAR_EFFECT = {"FL-8000": 0.0, "Sanibel": -4.5, "Tygress": -9.0,
                   "HM-1823": +3.0, "Sebring": -2.0}

TRANSPLANT = ["certified", "on-farm", "regional"]
TRANSPLANT_EFFECT = {"certified": 0.0, "on-farm": +5.0, "regional": +2.0}

# ------------------------------------------------------------------- true coefficients
# latent severity = intercept + sum(beta * standardized-ish feature) + effects + noise
TRUE = {
    "intercept": 45.0,
    "wetness_hours": 0.075,      # per cumulative hour of leaf wetness
    "mean_temp_c": 2.20,         # per degree C
    "max_temp_c": 0.0,           # NOT causal - only correlated with mean_temp_c
    "rain_mm": 0.025,            # per mm
    "inoculum_log": 4.50,        # per log10 unit of initial inoculum
    "soil_ph": 0.0,              # pure noise
    "dist_to_road_m": 0.0,       # pure noise
    "site_sd": 5.0,
    "year_sd": 7.0,
    "resid_sd": 9.0,
    "squash_midpoint": 45.0,     # latent value mapped to 50% severity
    "squash_k": 0.090,           # steepness of the logistic squash
}

SITE_RE = {s: v for s, v in zip(SITES, RNG.normal(0, TRUE["site_sd"], len(SITES)))}
YEAR_RE = {y: v for y, v in zip(YEARS, RNG.normal(0, TRUE["year_sd"], len(YEARS)))}


def squash(latent, midpoint, k):
    """Map an unbounded latent score onto (0, 100) with floor and ceiling pile-up.

    Severity is a bounded proportion: real ratings crowd near 0 in clean seasons and
    near 100 in blown-out plots. A logistic squash reproduces both, which is what makes
    the observed-vs-predicted diagnostic slide honest."""
    return 100.0 / (1.0 + np.exp(-k * (latent - midpoint)))


# ------------------------------------------------------------------ season-level table
rows = []
for year in YEARS:
    # season-level weather anchors, so plots within a site-year share conditions
    year_wet = RNG.normal(310, 55)
    year_temp = RNG.normal(24.5, 1.1)
    for site in SITES:
        site_wet = year_wet + RNG.normal(0, 25)
        site_temp = year_temp + RNG.normal(0, 0.6)
        for p in range(PLOTS_PER_SITE_YEAR):
            wetness = max(80.0, site_wet + RNG.normal(0, 18))
            mean_t = site_temp + RNG.normal(0, 0.5)
            # max temp is strongly correlated with mean temp but has NO causal effect
            max_t = 8.2 + 0.94 * mean_t + RNG.normal(0, 0.55)
            rain = max(0.0, RNG.normal(1.9, 0.6) * wetness * 0.55 + RNG.normal(0, 40))
            inoc = RNG.normal(3.1, 0.85)
            cult = RNG.choice(CULTIVARS, p=[0.30, 0.22, 0.18, 0.15, 0.15])
            tp = RNG.choice(TRANSPLANT, p=[0.45, 0.30, 0.25])
            ph = RNG.normal(6.4, 0.35)
            road = RNG.uniform(20, 900)

            latent = (
                TRUE["intercept"]
                + TRUE["wetness_hours"] * (wetness - 300)
                + TRUE["mean_temp_c"] * (mean_t - 24.5)
                + TRUE["rain_mm"] * (rain - 330)
                + TRUE["inoculum_log"] * (inoc - 3.1)
                + CULTIVAR_EFFECT[cult]
                + TRANSPLANT_EFFECT[tp]
                + SITE_RE[site]
                + YEAR_RE[year]
                + RNG.normal(0, TRUE["resid_sd"])
            )
            sev = squash(latent, TRUE["squash_midpoint"], TRUE["squash_k"])
            audpc = sev * RNG.normal(11.5, 1.1)  # crude AUDPC proxy in %-days

            rows.append(dict(
                year=year, site=site, county=COUNTY[site], plot=f"{site}-{year}-P{p+1}",
                cultivar=cult, transplant_source=tp,
                wetness_hours=round(wetness, 1), mean_temp_c=round(mean_t, 2),
                max_temp_c=round(max_t, 2), rain_mm=round(rain, 1),
                inoculum_log=round(inoc, 2), soil_ph=round(ph, 2),
                dist_to_road_m=round(road, 0),
                final_severity=round(sev, 2), audpc=round(audpc, 1),
            ))

season = pd.DataFrame(rows)

# a few realistic missing values in one covariate
miss = RNG.choice(season.index, size=int(0.03 * len(season)), replace=False)
season.loc[miss, "inoculum_log"] = np.nan

# --------------------------------------------------------------- weekly binary table
# target: did a scoutable infection event occur in this plot-week?
TRUE_LOGIT = {
    "intercept": -4.10,
    "wet_hours_week": 0.062,     # per hour of leaf wetness in the week
    "night_temp_c": 0.230,       # per degree C of mean night temperature
    "rain_mm_week": 0.011,
    "inoculum_log": 0.780,
    "days_after_transplant": 0.021,
    "plot_re_sd": 0.90,          # persistent plot susceptibility (microclimate, drainage,
                                 # edge effects). Constant within a plot -> this is what a
                                 # flexible model memorizes when rows from one plot land in
                                 # both the training and test folds.
    "cultivar": {"FL-8000": 0.0, "Sanibel": -0.55, "Tygress": -1.15,
                 "HM-1823": +0.40, "Sebring": -0.30},
}

PLOT_RE = {p: v for p, v in zip(season["plot"],
                                RNG.normal(0, TRUE_LOGIT["plot_re_sd"], len(season)))}

wrows = []
for _, r in season.iterrows():
    inoc = r["inoculum_log"] if not np.isnan(r["inoculum_log"]) else 3.1
    for wk in range(1, 13):
        dat = wk * 7
        wet_w = max(2.0, RNG.normal(r["wetness_hours"] / 12.0, 6.0))
        night_t = RNG.normal(r["mean_temp_c"] - 4.0, 1.3)
        rain_w = max(0.0, RNG.normal(r["rain_mm"] / 12.0, 9.0))
        z = (TRUE_LOGIT["intercept"]
             + TRUE_LOGIT["wet_hours_week"] * wet_w
             + TRUE_LOGIT["night_temp_c"] * (night_t - 20.0)
             + TRUE_LOGIT["rain_mm_week"] * rain_w
             + TRUE_LOGIT["inoculum_log"] * (inoc - 3.1)
             + TRUE_LOGIT["days_after_transplant"] * (dat - 42)
             + TRUE_LOGIT["cultivar"][r["cultivar"]]
             + PLOT_RE[r["plot"]]
             + 0.35 * SITE_RE[r["site"]] / TRUE["site_sd"]
             + 0.35 * YEAR_RE[r["year"]] / TRUE["year_sd"])
        p = 1.0 / (1.0 + np.exp(-z))
        wrows.append(dict(
            year=r["year"], site=r["site"], county=r["county"], plot=r["plot"],
            week=wk, days_after_transplant=dat, cultivar=r["cultivar"],
            wet_hours_week=round(wet_w, 1), night_temp_c=round(night_t, 2),
            rain_mm_week=round(rain_w, 1), inoculum_log=round(inoc, 2),
            true_prob=round(p, 4),
            infection_event=int(RNG.random() < p),
        ))

weekly = pd.DataFrame(wrows)

season.to_csv("data/bacterial_spot_season.csv", index=False)
weekly.drop(columns=["true_prob"]).to_csv("data/bacterial_spot_weekly.csv", index=False)
weekly.to_csv("data/bacterial_spot_weekly_withtruth.csv", index=False)

truth = {
    "description": "Ground truth for the synthetic bacterial spot dataset.",
    "season_model": {
        "form": "severity = ceiling * (1 - exp(-k * latent)); latent = intercept + sum(beta*(x - center)) + cultivar + transplant + site_RE + year_RE + noise",
        "centers": {"wetness_hours": 300, "mean_temp_c": 24.5, "rain_mm": 330, "inoculum_log": 3.1},
        "coefficients": {k: v for k, v in TRUE.items()},
        "cultivar_effect": CULTIVAR_EFFECT,
        "transplant_effect": TRANSPLANT_EFFECT,
        "site_random_effects": {k: round(v, 3) for k, v in SITE_RE.items()},
        "year_random_effects": {str(k): round(v, 3) for k, v in YEAR_RE.items()},
        "decoys": ["max_temp_c (r~0.93 with mean_temp_c, zero causal effect)",
                   "soil_ph (noise)", "dist_to_road_m (noise)"],
    },
    "weekly_model": {k: v for k, v in TRUE_LOGIT.items()},
    "weekly_note": ("A plot-level random effect (sd 0.90 on the logit) is constant within "
                    "each plot. inoculum_log is also constant within a plot, so it acts as a "
                    "plot fingerprint: a flexible model can memorize plot susceptibility "
                    "through it. That is why a random row-wise split leaks and a "
                    "leave-one-plot-out split does not."),
}
with open("data/truth.json", "w") as f:
    json.dump(truth, f, indent=2)

# ------------------------------------------------------------------------ quick report
print("season:", season.shape, "| weekly:", weekly.shape)
print("severity  mean %.1f  sd %.1f  max %.1f  | %% at >95: %.1f"
      % (season.final_severity.mean(), season.final_severity.std(),
         season.final_severity.max(), 100 * (season.final_severity > 95).mean()))
print("corr(mean_temp, max_temp) = %.3f" % season[["mean_temp_c", "max_temp_c"]].corr().iloc[0, 1])
print("infection event prevalence: %.3f" % weekly.infection_event.mean())
print("severity by year sd = %.2f  (year effect present)" % season.groupby("year").final_severity.mean().std())
