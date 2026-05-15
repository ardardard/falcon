# Falcon — EMWA Variance Tracker

Probabilistic dove-location tracker for the [CrunchDAO Falcon](https://hub.crunchdao.com/competitions/falcon) competition. It uses a two-Gaussian mixture with EWMA variance, velocity extrapolation, rolling volatility, direction-change widening, and falcon-proximity scaling.

## Files

| File | Description |
|------|-------------|
| `emwavartracker.py` | `EMWAVarTracker` class (main model) |
| `falcon.py` | Submission entry point — creates `tracker = EMWAVarTracker()` |
| `requirements.txt` | Python dependencies |

## Requirements

- Python 3.10+ (3.12 recommended)
- Packages: `birdgame`, `numpy`, `river`

## Setup

Open a terminal in this folder:

```bash
cd C:\Users\ardaf\OneDrive\Masaüstü\falcon
```

Install dependencies:

```bash
py -3 -m pip install -r requirements.txt
```

## Run locally

Compare your tracker against the built-in benchmark on remote test data (~50k rows by default):

```bash
py -3 falcon.py
```

Or run the tracker module directly:

```bash
py -3 emwavartracker.py
```

Both call `EMWAVarTracker().test_run(live=False, step_print=1000)`, which prints mean log-likelihood vs the benchmark every 1000 steps.

### Use live streaming data

In `falcon.py`, change the `test_run` call to:

```python
tracker.test_run(live=True, step_print=1000)
```

Live mode needs network access and valid birdgame live-data credentials.

### Quick import check

```bash
py -3 -c "from emwavartracker import EMWAVarTracker; print(EMWAVarTracker())"
```

## Submit to CrunchDAO

1. Open [Falcon → Submit](https://hub.crunchdao.com/competitions/falcon/submit/files) (or notebook submit if you prefer).
2. Upload at minimum:
   - `falcon.py`
   - `emwavartracker.py`
   - `requirements.txt`
3. Ensure your submission exposes a `TrackerBase` subclass with `tick()` and `predict()` — `falcon.py` already instantiates `EMWAVarTracker`.

Optional: use the Crunch CLI from the competition page:

```bash
py -3 -m pip install crunch-cli --upgrade
crunch setup falcon <your-model-name> --token <token> "C:\Users\ardaf\OneDrive\Masaüstü\falcon"
cd "C:\Users\ardaf\OneDrive\Masaüstü\falcon"
crunch test
crunch push --message "EMWA variance tracker"
```

## Tuning

Edit attributes on `EMWAVarTracker` in `emwavartracker.py`, for example:

| Parameter | Default | Role |
|-----------|---------|------|
| `weights` | `[0.92, 0.08]` | Core vs tail mixture weights |
| `winsor_mult` | `2.5` | Core variance winsorization |
| `velocity_blend` | `0.4` | Mean extrapolation strength (`0` = current position only) |
| `vol_mult_cap` | `1.25` | Max uncertainty scaling from rolling vol |

## Payload format

Each `tick()` receives a dict like:

```python
{
    "time": 53952.149,
    "dove_location": 2200.16,
    "falcon_location": 2200.15,
    "falcon_id": 72,
    "falcon_wingspan": 0.2014,
}
```

`predict()` returns a Gaussian mixture in [densitypdf](https://github.com/microprediction/densitypdf) format.

## Links

- [Birdgame docs / examples](https://github.com/microprediction/birdgame)
- [Falcon quickstarter intro](https://forum.crunchdao.com/t/falcon-introduction-to-notebook-quickstarters/1060)
