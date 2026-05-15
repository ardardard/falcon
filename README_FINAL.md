# Falcon Tracker (BSDSA Team 2)

EMWA variance mixture tracker for the [Bird Game](https://hub.crunchdao.com/competitions/falcon) dove-location forecasting challenge, built on the `birdgame` framework.

## Project layout

| Path | Description |
|------|-------------|
| `emwavartracker.py` | Modified version of the base model "Variance Tracker" |
| `falcon.py` | Submission entry point that loads the optimized tracker |
| `requirements.txt` | Python dependencies |

## Setup

Use Python 3.10+ (3.12 recommended). Create a virtual environment if you like, then install dependencies from the repo root:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`birdgame` is installed from PyPI and brings in its own dependencies, including `densitypdf` for scoring. You do not need to copy the full `birdgame-main` folder into this repository as long as `birdgame` is listed in `requirements.txt`.

## Running a tracker

Each model file defines a tracker class and can be executed directly. From the repository root:

```bash
python emwavartracker.py
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

path = Path("emwavartracker.py")
mod = SourceFileLoader("tracker", str(path)).load_module()

tracker = mod.EMWAVarTracker()
tracker.test_run(
    live=False,       # False = remote CSV test set; True = live stream
    step_print=1000,  # print interim scores every N steps
    max_rows=10000,   # optional cap for quicker experiments
)
```

You can either run the tracker file directly or import it from another script for experiments.

## Model: `EMWAVarTracker`

Estimates core and tail variance of dove displacements with exponentially weighted statistics, then predicts a two-Gaussian mixture centered at the current location, with optional velocity extrapolation in the optimized build.

The optimized version adds rolling-volatility scaling, direction-change and falcon-proximity widening, and tuned mixture weights (`0.92` / `0.08`) and scale calibration for higher `test_run` scores than the reference tracker.

## Notes

- Predictions must follow the [`density_pdf`](https://github.com/microprediction/densitypdf) mixture format returned by `predict()`.
- During platform submission, models may still use a separate warmup window; local `test_run` scores every tick unless your tracker returns `None` from `predict()`.
- For animated comparison plots, see `TrackerBase.test_run_animated()` in the birdgame documentation.
