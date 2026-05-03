# Falcon Competition — Baseline Models

Two baseline trackers for the CrunchDAO Falcon competition.

Setup: `pip install birdgame`

---

# Model 1: EWMA Variance Tracker

## What this model does

This tracker predicts where the dove will be in the near future by looking at how much the dove has been moving recently. It doesn't try to guess a specific direction; it just says "the dove is probably near where it is now" and adjusts how confident it is based on how jumpy the dove's movements have been.

It tracks the variance (how spread out the movements are) of the dove's position changes over time using an exponentially weighted moving average, which basically means recent movements matter more than older ones. At each tick, we look at how much the dove moved since the last observation and feed that into two separate variance trackers: one that clips extreme values to focus on normal movements, and another that scales them up to capture big jumps. The final prediction is a mixture of two Gaussians: a narrow one with 95% weight for the normal case, and a wider one with 5% weight for rare large moves. Both are centered on where the dove currently is.

---

## Code breakdown

### Imports

```python
import math
import numpy as np
from birdgame.trackers.trackerbase import TrackerBase
from birdgame.stats.fewvar import FEWVar
from birdgame import HORIZON
```

- `math` and `numpy` are just for basic math operations (`sqrt`, `clip`).
- `TrackerBase` is the base class every tracker needs to inherit from. It gives us the `tick()` / `predict()` interface and the quarantine system.
- `FEWVar` is a helper class from the birdgame library that computes an Exponentially Weighted Moving Average of variance. Basically it tracks how spread out the values are, but gives more importance to recent values.
- `HORIZON` is how many seconds ahead we need to predict (it's set to 3).

### The class and `__init__`

```python
class EMWAVarTracker(TrackerBase):

    def __init__(self, horizon=HORIZON):
        super().__init__(horizon)
        self.current_x = None
        self.ewa_dx_core = FEWVar(fading_factor=0.0001)
        self.ewa_dx_tail = FEWVar(fading_factor=0.0001)
```

- `super().__init__(horizon)` sets up the base tracker with the quarantine system.
- `self.current_x` stores the dove's most recent position.
- `self.ewa_dx_core` tracks the variance of **normal** movements. The `fading_factor=0.0001` means it adapts slowly — old observations lose influence gradually, so the estimate is smooth and stable.
- `self.ewa_dx_tail` tracks the variance of **extreme** movements. Same fading factor, but it gets fed scaled-up values (more on that below).

### The `tick()` method

```python
def tick(self, payload, performance_metrics=None):
    x = payload['dove_location']

    self.add_to_quarantine(payload['time'], x)
    prev_x = self.pop_from_quarantine(payload['time'])

    if prev_x is not None:
        self.ewa_dx_core.update(np.clip(x - prev_x, -2, 2))
        self.ewa_dx_tail.update(2 * (x - prev_x))

    self.current_x = x
```

This gets called every time new data comes in. Here's what each line does:

1. **`x = payload['dove_location']`** — grab the dove's current position from the incoming data.

2. **`self.add_to_quarantine(payload['time'], x)`** — put the current observation into "quarantine." This is a waiting room. The observation won't be available for training until `HORIZON` seconds have passed, so we don't accidentally use future information when training.

3. **`prev_x = self.pop_from_quarantine(payload['time'])`** — check if any quarantined observation has matured (i.e., enough time has passed). If yes, we get it back as `prev_x`. If nothing is ready yet, `prev_x` is `None`.

4. **`if prev_x is not None:`** — if we have a valid previous observation, we can compute how much the dove moved:
   - `x - prev_x` is the movement (change in position).
   - **Core update:** `np.clip(x - prev_x, -2, 2)` clips the movement to the range [-2, 2]. This prevents extreme outliers from messing up the "normal" variance estimate. We feed this to `ewa_dx_core`.
   - **Tail update:** `2 * (x - prev_x)` scales the movement by 2x, amplifying it. This makes the tail variance estimate larger, so the wide Gaussian in our prediction covers more extreme scenarios.

5. **`self.current_x = x`** — save the current position for use in `predict()`.

### The `predict()` method

```python
def predict(self):
    if self.ewa_dx_core.get() == 0.0:
        return None

    return {
        "type": "mixture",
        "components": [
            {
                "density": {
                    "type": "builtin",
                    "name": "norm",
                    "params": {
                        "loc": self.current_x,
                        "scale": math.sqrt(self.ewa_dx_core.get())
                    }
                },
                "weight": 0.95
            },
            {
                "density": {
                    "type": "builtin",
                    "name": "norm",
                    "params": {
                        "loc": self.current_x,
                        "scale": math.sqrt(self.ewa_dx_tail.get())
                    }
                },
                "weight": 0.05
            }
        ]
    }
```

1. **Warmup check:** If `ewa_dx_core.get() == 0.0`, we haven't seen enough data yet to estimate variance, so we return `None`. The competition doesn't penalize you during warmup.

2. **The prediction** is a mixture of two normal distributions:
   - **Component 1 (95% weight):** A Gaussian centered on the dove's current position (`loc`), with a standard deviation based on the core variance (`scale`). This is the "normal case" — most of the time the dove moves a small amount.
   - **Component 2 (5% weight):** Same center, but wider — the scale comes from the tail variance. This covers the case where the dove makes a big unexpected jump.

3. **Why `math.sqrt`?** Because `FEWVar.get()` returns the *variance* (average of squared deviations), but the Gaussian `scale` parameter needs the *standard deviation* (square root of variance).

4. **Why a mixture?** A single Gaussian might be too narrow for big jumps or too wide for normal ticks. By blending a narrow and wide one, we hedge: 95% of the time we bet the dove moves normally, but 5% of the time we allow for a big jump. This improves the log-likelihood score because we're less likely to assign near-zero probability to a surprising outcome.

### Running the benchmark

```python
if __name__ == "__main__":
    tracker = EMWAVarTracker()
    tracker.test_run(live=False, step_print=1000)
```

This runs the tracker against saved test data and prints the log-likelihood score every 1000 steps. Use `live=True` to test against the live data stream instead.

---

### Parameters you can tune

| Parameter | Current value | What it does |
|-----------|--------------|--------------|
| `fading_factor` | 0.0001 | How fast old data loses influence. Higher = adapts faster but noisier. |
| Core weight | 0.95 | How much we trust the "normal movement" Gaussian. |
| Tail weight | 0.05 | How much we trust the "big jump" Gaussian. |
| Clip range | [-2, 2] | What counts as a "normal" movement for the core variance. |
| Tail scaling | 2x | How much we amplify movements for the tail variance. |

### Strengths

- Very fast — well under the 50ms inference limit
- Simple to understand and debug
- Adapts to changing volatility automatically through the exponential weighting
- The two-Gaussian mixture is a nice safety net against unexpected big moves
- Good benchmark to compare other models against

### Limitations

- Doesn't actually learn any relationship between input and output — it just tracks how much things move
- Always predicts the dove stays where it is now (the center of both Gaussians is always the current position)
- Completely ignores the falcon data (`falcon_location`, `falcon_id`, `falcon_wingspan`)
- Can't incorporate new features easily — there's no place to plug in extra inputs

---

# Model 2: Quantile Regression Tracker

## What this model does

This tracker learns from the data by watching where the dove was and where it ended up, and over time it picks up on the relationship between the two. It trains three models at the same time, each targeting a different percentile: the 5th (low estimate), the 50th (most likely outcome), and the 95th (high estimate). The 50th percentile becomes the center of our prediction, and the spread between the 5th and 95th tells us how uncertain we should be: a wider spread means less confidence, a tighter one means more. What makes this useful compared to the EWMA tracker is that it's actually learning, not just tracking how much things move. And it's built in a way where we can easily feed in extra information later, like the dove's speed or how close a falcon is, without having to restructure anything.
The prediction is a single Gaussian centered on the median estimate.

---

## Code breakdown

### Imports

```python
import numpy as np
from river import linear_model, preprocessing, optim
from birdgame.trackers.trackerbase import TrackerBase
from birdgame import HORIZON
```

- `numpy` for basic math.
- `river` is a library for online (streaming) machine learning. We use it for linear regression that updates one observation at a time instead of needing a whole dataset.
- `TrackerBase` and `HORIZON` are the same as in the EWMA tracker.

### The class and `__init__`

```python
class QuantileRegressionRiverTracker(TrackerBase):

    def __init__(self, horizon=HORIZON):
        super().__init__(horizon)
        self.current_x = None
        self.is_warm = False

        self.models = {}
        for alpha in [0.05, 0.5, 0.95]:
            scale = preprocessing.StandardScaler()
            learn = linear_model.LinearRegression(
                intercept_lr=0,
                optimizer=optim.SGD(0.005),
                loss=optim.losses.Quantile(alpha=alpha)
            )
            model = scale | learn
            model = preprocessing.TargetStandardScaler(regressor=model)
            self.models[f"q {alpha:.2f}"] = model
```

- `self.is_warm` tracks whether we've received enough data to start predicting.
- `self.models` is a dictionary holding three separate models, one per quantile:
  - `alpha=0.05` → predicts the 5th percentile (low end, "dove probably won't go below this")
  - `alpha=0.5` → predicts the 50th percentile (median, our best guess)
  - `alpha=0.95` → predicts the 95th percentile (high end, "dove probably won't go above this")
- Each model is a pipeline built with the `|` operator:
  - `StandardScaler()` normalizes the input features so the model trains more stably.
  - `LinearRegression(...)` does the actual learning. The `loss=Quantile(alpha=...)` is what makes each model target a different percentile instead of the mean.
  - `TargetStandardScaler` normalizes the target values too.
- `intercept_lr=0` means we don't learn a separate intercept term.
- `SGD(0.005)` is stochastic gradient descent with a learning rate of 0.005.

### The `tick()` method

```python
def tick(self, payload, performance_metrics=None):
    x = payload['dove_location']

    self.add_to_quarantine(payload['time'], x)
    prev_x = self.pop_from_quarantine(payload['time'])

    if prev_x is not None:
        for m in self.models.values():
            m.learn_one({"x": prev_x}, x)
        self.is_warm = True

    self.current_x = x
```

1. **`x = payload['dove_location']`** — get the current dove position.

2. **Quarantine** works the same as in EWMA — we wait `HORIZON` seconds before using an observation for training.

3. **`if prev_x is not None:`** — once we have a matured observation:
   - `m.learn_one({"x": prev_x}, x)` trains each of the 3 models. The input is the previous position (`prev_x`) and the target is the current position (`x`). So the model is learning: "given where the dove was, where is it now?"
   - The input is a dictionary `{"x": prev_x}` — this is where you'd add more features later (like `{"x": prev_x, "speed": speed, "falcon_dist": dist}`).
   - `self.is_warm = True` — we've seen at least one training example, so we can start predicting.

4. **`self.current_x = x`** — save the current position.

### The `predict()` method

```python
def predict(self):
    if not self.is_warm or self.current_x is None:
        return None

    x_mean = self.models["q 0.50"].predict_one({"x": self.current_x})
    y_lower = self.models["q 0.05"].predict_one({"x": self.current_x})
    y_upper = self.models["q 0.95"].predict_one({"x": self.current_x})

    scale = max(abs(y_upper - y_lower) / 3.29, 1e-6)

    return {
        "type": "mixture",
        "components": [
            {
                "density": {
                    "type": "builtin",
                    "name": "norm",
                    "params": {
                        "loc": x_mean,
                        "scale": scale
                    }
                },
                "weight": 1
            }
        ]
    }
```

1. **Warmup check:** If we haven't trained yet or don't have a current position, return `None`.

2. **Three predictions:**
   - `x_mean` — the median model's prediction (our best guess for where the dove will be)
   - `y_lower` — the 5th percentile (lower bound)
   - `y_upper` — the 95th percentile (upper bound)

3. **`scale = max(abs(y_upper - y_lower) / 3.29, 1e-6)`** — this converts the percentile range into a standard deviation. Why 3.29? In a normal distribution, the distance from the 5th to 95th percentile is about 3.29 standard deviations (2 × 1.645). The `max(..., 1e-6)` prevents the scale from being zero.

4. **The prediction** is a single Gaussian centered on the median estimate with the computed scale. Unlike the EWMA tracker which uses two Gaussians, this one uses just one because the uncertainty is already learned from data.

### Running the benchmark

```python
if __name__ == "__main__":
    tracker = QuantileRegressionRiverTracker()
    tracker.test_run(live=False, step_print=1000)
```

Same as EWMA — runs against test data and prints scores.

---

### Parameters you can tune

| Parameter | Current value | What it does |
|-----------|--------------|--------------|
| Learning rate | 0.005 | How fast the model updates. Higher = learns faster but less stable. |
| Quantiles | 0.05, 0.50, 0.95 | Which percentiles to estimate. |
| `intercept_lr` | 0 | Learning rate for intercept. Set to 0 so we don't learn one. |

### Strengths

- Actually learns from data — builds a mapping from current position to future position
- Easy to extend with new features: just add more keys to the `{"x": prev_x}` dictionary in `tick()` and `predict()`
- Uncertainty is data-driven (based on learned percentile spread, not just variance tracking)
- Still fast enough for the 50ms limit
- Good starting point for the team to build on

### Limitations

- Currently only uses one feature (current position) — doesn't use falcon data yet
- Linear model, so it can't capture complex non-linear patterns
- Single Gaussian output means it doesn't handle bimodal situations (where the dove might go one of two very different directions)
- Needs a few observations to warm up before predictions are reliable
