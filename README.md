
# AI Safety Guardian

**Proactive safety AI for smart cities — built on CAMARA Open Gateway APIs (Location, Geofencing, QoD, Device Status).**

Most safety apps only react after you press a panic button. AI Safety Guardian watches ambient signals — location pattern, geofence risk zones, movement, and device connectivity — to detect danger *before* you have to ask for help.

Switch on travel mode before heading out, and if you stop moving for longer than expected, it checks in on you. No response, and it escalates: notifying your trusted circle, reserving network quality for emergency communication, and alerting emergency services — all with a one-tap "I'm safe" to cancel at any point.

Built for tourists, women traveling alone, and teenagers. Available as an in-app dashboard and a lock screen / home screen widget for at-a-glance status.

**APIs used:** Geofencing · Location Retrieval · Quality on Demand (QoD) · Device Status

# AI Safety Guardian — ML/NN Integration

Notes on where machine learning fits into the existing rule-based risk engine for **AI Safety Guardian**, a proactive urban safety companion (CAMARA APIs: Geofencing, Location, QoD, Device Status).

## Why ML?

The current design uses fixed thresholds (e.g. "stationary > 5 min → escalate"). This doesn't generalize across users and doesn't improve from real usage. ML layers on top to fix both.

## Proposed Models

| # | Model | Replaces / Upgrades | Feasibility |
|---|-------|----------------------|--------------|
| 1 | **Learned Risk Scoring Engine** — small NN or gradient-boosted tree outputting a 0–1 risk score | Hardcoded risk thresholds | 🟢 High |
| 2 | **Movement Pattern Classification** — 1D-CNN/LSTM over IMU data (walking, idle, erratic, fall signature) | Movement Detection / Stationary Check-In | 🟡 Medium |
| 3 | **Personalized Anomaly Detection** — per-user autoencoder flagging deviation from normal routine | New layer on the Risk Engine | 🟡 Medium |
| 4 | **Spatiotemporal Risk Prediction** — dynamic risk heatmap by time + location | Static Geofenced Risk Zones | 🔴 Low (needs real city incident data) |
| 5 | **Audio Distress Detection** — on-device CNN classifying shouting/distress sounds | Silent Mode trigger | 🟡 Medium (stretch goal) |

## Build Priority (hackathon timeline)

1. Risk Scoring Engine — synthetic training data, swap in for rule-based scorer
2. Movement Classification — pretrained HAR model (UCI-HAR / WISDM / MobiAct), fine-tuned, on-device via TFLite/Core ML
3. Anomaly Detection — demo with simulated user history
4. Spatiotemporal Prediction — pitch as roadmap item, don't build
5. Audio Distress Detection — only if time remains

## Training Data Sources

- **Risk Engine**: synthetic journey simulations + public mobility/incident datasets
- **Movement**: UCI-HAR, WISDM, MobiAct (includes fall data)
- **Anomaly Detection**: per-user history, cold-started from population averages
- **Audio**: AudioSet (distress subset), ESC-50

## Suggested Stack

- Training: Python, scikit-learn / PyTorch, pandas
- On-device inference: TensorFlow Lite (Android), Core ML (iOS)
- Serving: FastAPI alongside existing backend
- Experiment tracking: Weights & Biases / MLflow.

## Notes

- All on-device models run **only after explicit user opt-in** (Silent Mode, Movement Detection), consistent with the product's "calm by default, no daily intrusion" principle.
- Full technical write-up: see `AI_Safety_Guardian_ML_Integration.pdf`