# Falcon Tracker (BSDSA Team 2)

EMWA variance mixture tracker for the [Bird Game](https://hub.crunchdao.com/competitions/falcon) dove-location forecasting challenge, built on the `birdgame` framework.

## Project layout

| Path | Description |
|------|-------------|
| `base_model/emwa_var_tracker` | Reference implementation (starting point) |
| `optimized_model/emwa_var_tracker_optimized` | Tuned version for higher `test_run` scores |
| `falcon.py` | Submission entry point (loads the optimized tracker) |
| `requirements.txt` | Python dependencies |

## Setup

Use Python 3.10+ (3.12 recommended). Create a virtual environment if you like, then install dependencies from the repo root:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`birdgame` is installed from PyPI and brings in its own dependencies (including `densitypdf` for scoring).

## Running a tracker

Each model file defines a tracker class and can be executed directly. From the repository root:

```bash
python optimized_model/emwa_var_tracker_optimized
python base_model/emwa_var_tracker
python falcon.py
```

By default, scripts call `test_run(live=False)`, which streams historical test data and prints log-likelihood scores. **Higher scores are better.**

### `test_run` behavior

`test_run` (from `birdgame.trackers.trackerbase.TrackerBase`):

1. Feeds time-series payloads (`time`, `dove_location`, falcon fields, …) into your tracker.
2. Scores each density prediction with `densitypdf` when the prediction horizon elapses.
3. Compares your mean log-likelihood to the built-in **EMWA variance benchmark** and prints whether yours is better or worse.

Options you can pass when invoking from your own script:

```python
from importlib.machinery import SourceFileLoader
from pathlib import Path

path = Path("optimized_model/emwa_var_tracker_optimized")
mod = SourceFileLoader("tracker", str(path)).load_module()

tracker = mod.EMWAVarTracker()
tracker.test_run(
    live=False,       # False = remote CSV test set; True = live stream
    step_print=1000,  # print interim scores every N steps
    max_rows=10000,   # optional cap for quicker experiments
)
```

To import a tracker module whose filename has no `.py` extension, either run it as a script (as above) or load it with `importlib`.

## Model: `EMWAVarTracker`

Estimates core and tail variance of dove displacements with exponentially weighted statistics, then predicts a two-Gaussian mixture centered at the current location (with optional velocity extrapolation in the optimized build).

The optimized version adds rolling-volatility scaling, direction-change and falcon-proximity widening, and tuned mixture weights (`0.92` / `0.08`) and scale calibration for higher `test_run` scores than the reference tracker.

## Notes

- Predictions must follow the [`density_pdf`](https://github.com/microprediction/densitypdf) mixture format returned by `predict()`.
- During platform submission, models may still use a separate warmup window; local `test_run` scores every tick unless your tracker returns `None` from `predict()`.
- For animated comparison plots, see `TrackerBase.test_run_animated()` in the birdgame documentation.
