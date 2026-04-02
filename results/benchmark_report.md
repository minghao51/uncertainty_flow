================================================================================
COMPREHENSIVE BENCHMARK COMPARISON REPORT
================================================================================

RESULTS BY DATASET
--------------------------------------------------------------------------------

exchange_rate (Finance)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  Best Winkler @ 90%: conformal-forecaster (0.3613)
  Best Coverage @ 90%: conformal-forecaster (0.8795)
  Fastest Training: naive-forecast (0.000s)

weather (Climate)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  Best Winkler @ 90%: random-forest (0.0585)
  Best Coverage @ 90%: linear-regression (0.9340)
  Fastest Training: naive-forecast (0.000s)

electricity (Energy)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  Best Winkler @ 90%: quantile-forest (306.0097)
  Best Coverage @ 90%: quantile-forest (0.9280)
  Fastest Training: naive-forecast (0.000s)


OVERALL MODEL RANKINGS (by average Winkler @ 90%)
--------------------------------------------------------------------------------

  1. quantile-forest: 102.2600
  2. random-forest: 108.7778
  3. conformal-forecaster: 114.1700
  4. conformal-regressor: 122.1890
  4. gradient-boosting: 122.1890
  5. linear-regression: 283.6228
  6. ridge-regression: 283.8519
  7. moving-average: 458.4653
  8. naive-forecast: 481.4650


COMPARISON BY MODEL CATEGORY
--------------------------------------------------------------------------------

Uncertainty Flow Models:
    quantile-forest: cov=0.823, sharp=81.5945, wink=102.2600, time=0.255s
    conformal-forecaster: cov=0.792, sharp=103.7836, wink=114.1700, time=0.287s
    conformal-regressor: cov=0.753, sharp=108.5528, wink=122.1890, time=0.302s

Regression Baselines:
    random-forest: cov=0.799, sharp=98.1789, wink=108.7778, time=0.108s
    gradient-boosting: cov=0.753, sharp=108.5528, wink=122.1890, time=0.356s
    linear-regression: cov=0.802, sharp=274.6183, wink=283.6228, time=0.024s
    ridge-regression: cov=0.779, sharp=275.1602, wink=283.8519, time=0.023s

Simple Baselines:
    moving-average: cov=0.345, sharp=239.2003, wink=458.4653, time=0.002s
    naive-forecast: cov=0.353, sharp=256.1676, wink=481.4650, time=0.000s


KEY FINDINGS
--------------------------------------------------------------------------------

1. Best Uncertainty Flow Model: quantile-forest (avg Winkler @ 90%: 102.2600)
2. Best Baseline Model: random-forest (avg Winkler @ 90%: 108.7778)

COVERAGE ANALYSIS
--------------------------------------------------------------------------------

  Models ranked by average coverage @ 90%:
    quantile-forest: cov90=0.823 (dev=0.077), cov80=0.767 (dev=0.033)
    linear-regression: cov90=0.802 (dev=0.098), cov80=0.761 (dev=0.039)
    random-forest: cov90=0.799 (dev=0.101), cov80=0.701 (dev=0.099)
    conformal-forecaster: cov90=0.792 (dev=0.108), cov80=0.641 (dev=0.159)
    ridge-regression: cov90=0.779 (dev=0.121), cov80=0.725 (dev=0.075)
    conformal-regressor: cov90=0.753 (dev=0.147), cov80=0.654 (dev=0.146)
    gradient-boosting: cov90=0.753 (dev=0.147), cov80=0.654 (dev=0.146)
    naive-forecast: cov90=0.353 (dev=0.547), cov80=0.314 (dev=0.486)
    moving-average: cov90=0.345 (dev=0.555), cov80=0.275 (dev=0.525)

================================================================================