# Movement Tracking Analysis & Feature Engineering Report

## 1. Introduction
This section analyzes the existing tracking models and proposes feature engineering improvements to enhance predictive performance. The goal is to enable the model to better capture movement dynamics, including **speed**, **direction changes**, and **temporal patterns**.

---

## 2. EWMA Variance Tracker
The Exponentially Weighted Moving Average (EWMA) tracker uses only the current dove location and the previous dove location. 

### Core Mechanics
*   **Primary Signal:** dx = current\_dove\_location - previous\_dove\_location
*   **Variance Estimates:** It maintains two exponentially weighted estimates:
    1.  **Clipped “Core”:** For normal movement.
    2.  **Scaled “Tail”:** For larger jumps.
*   **Prediction:** A two-Gaussian mixture centered at the current dove location.

### SWOT Analysis
| Strengths | Limitations |
| :--- | :--- |
| Very fast and computationally efficient. | Does not learn or predict movement direction. |
| Simple and stable; provides solid baseline. | No explicit modeling of velocity or acceleration. |
| Handles large jumps better than a single Gaussian. | Ignores falcon-related inputs. |
| Uncertainty adapts to recent volatility. | Always centers on current position (limited forecasting). |

---

## 3. Quantile Regression Tracker
The quantile regression tracker trains three online linear models to predict the **5th**, **50th (median)**, and **95th** percentiles.

*   **Current Input:** `{"x": prev_x}` (Maps previous position → current position).
*   **Method:** The median acts as the central estimate, while the 5th and 95th percentiles define the uncertainty range.

### SWOT Analysis
| Strengths | Limitations |
| :--- | :--- |
| Learns directly from data. | Uses only a single input feature. |
| Supports additional features via input dictionary. | Cannot capture temporal dynamics (speed/trends). |
| Produces data-driven uncertainty estimates. | No rolling statistics or directional logic. |
| Efficient enough for online learning. | Highly dependent on feature quality. |

---

## 4. Model Improvement Focus
The strongest candidate for improvement is the quantile regression tracker, since the EWMA tracker is mainly an uncertainty tracker and has no natural place to learn richer relationships. The quantile regression tracker already has a machine learning pipeline and can be improved directly by replacing the weak one-feature input with a richer feature set.
Proposed Feature Improvements

1. Temporal features

2. Kinematic features

3. Directional features

4. Rolling statistics

5. Falcon-related features

### Proposed Feature Improvements

#### 1. Temporal Features
Incorporating lagged positions provides short-term memory: 
*   x\_lag\_1, x\_lag\_2, x\_lag\_3
*   *Benefit:* Helps detect trends, oscillations, and reversals.
These help the model infer whether the dove is moving steadily, oscillating, or reversing direction.
#### 2. Kinematic Features
Capturing basic motion dynamics:
*   **Displacement:** dx = current\_x - previous\_x
*   **Acceleration:** accel = current\_dx - previous\_dx
*   **Absolute Acceleration:** |accel|
These could be useful when the dove changes movement intensity.

#### 3. Directional Features
*   **Direction:** sign(dx)
*   **Direction Change:** sign(current\_dx) \neq sign(previous\_dx)
These help detect reversals and oscillation.

#### 4. Rolling Statistics
Summarizing behavior over a sliding window:
*   `rolling_mean_x`, `rolling_std_x`, `rolling_range_x`
*   `rolling_mean_dx`, `rolling_std_dx`, `rolling_mean_abs_dx`
These capture volatility, stability, and whether the dove has recently been moving a lot or staying still.

#### 5. Falcon-Related Features (External Influences)
*   Falcon location and relative distance: $falcon\_loc - dove\_loc$
*   `falcon_wingspan`, `falcon_id`

---
## 5. Expected Benefits
the expected benefits from the improved tracker are:
- More accurate prediction of future position through movement extrapolation
- Improved uncertainty estimation combining learned quantiles and volatility features
- Better handling of directional changes via acceleration and direction features
- Increased adaptability across different movement regimes
- Potential integration of falcon-related behavior if available
