# Research: AURA — Explainable Women-Safety Surveillance

## AURA overview
In one line: AURA is an explainable four-stage AI video pipeline (YOLOv11 detection → tracking with an interpretable stalking score → R(2+1)D-18 violence recognition → XGBoost risk fusion with SHAP explanations) for women's safety surveillance, where every alert shows the evidence behind it.

AURA is my first-author research on an explainable four-stage video pipeline for women's safety surveillance. The manuscript is titled "An Interpretable Stalking Score for Women's Safety Surveillance within an Explainable Four-Stage Video Pipeline". It has been submitted to the International Journal of System Assurance Engineering and Management (Springer): https://link.springer.com/journal/13198 I keep the GitHub repository README-only while the paper is under review, to respect journal confidentiality.

The core idea: one person persistently following another is the behaviour that most often comes before harassment in public spaces, yet I could not find any system in the literature that measures it. So I defined an interpretable stalking score for a pair of tracked pedestrians, built from five components an operator can read separately: spatial proximity, duration of co-occurrence, agreement of heading, correlation of walking speed, and threat evidence carried forward from the detector. I deliberately kept it decomposed instead of collapsing it into one opaque probability, because an operator needs to know why a pair was flagged before acting.

## AURA architecture (four stages)
1. **Detection:** YOLOv11 detects persons, abusive interactions and weapons.
2. **Tracking and stalking score:** a Kalman filter with two-tier intersection-over-union (IoU) association maintains tracks, and every co-occurring pair is scored for following behaviour.
3. **Violence recognition:** an R(2+1)D-18 backbone with a bidirectional LSTM and attention estimates a violence probability from 16-frame clips.
4. **Risk fusion and explanation:** XGBoost fuses six behavioural and contextual features, and SHAP (Shapley) attributions are attached to every alert, so each alert can be traced to the evidence that produced it.

Every stage emits a logged intermediate signal, so a wrong alert can be traced to its origin: a missed detection, an identity switch, an over-sensitive stalking score, or a misread clip.

## AURA results
Each stage was evaluated separately on public data (COCO 2017, MOT17, RWF-2000 and a Roboflow women-safety dataset), with all experiments run on free Kaggle GPUs:
- The YOLOv11s detector reaches mAP@50 = 0.576 on a disjoint validation split.
- The violence module reaches 87.0% accuracy and AUC = 0.946 on RWF-2000.
- Detection with tracking runs in real time at 30.5 frames per second on a single NVIDIA T4 GPU.
- Stress test: on five MOT17 sequences containing no stalking at all, the score raised 121 false alerts. I report and diagnose that false-alarm behaviour rather than hide it. Because each alert carried its five components, I could see that none of the 121 alerts involved threat evidence and all came in through the proximity gate. For me that is the most useful result in the paper: interpretability is not only a feature for end users, it is also a debugging tool for developers.

## AURA limitations and ethics
- Limitations I state openly: weapon detection is weak (AP@50 0.315); the stalking score's pixel threshold is uncalibrated because no labelled following data exists; the fusion stage has so far been validated only on simulated features; and the full pipeline has not yet been validated end to end on labelled field footage.
- Ethics: the pipeline performs no facial recognition and infers no gender, track IDs exist only within a session, and alerts are advisory for a human operator rather than automated enforcement. A score that measures following could itself be misused for tracking people, so any deployment must sit under institutional governance, access control and independent oversight, and comply with India's Digital Personal Data Protection Act, 2023.
- Next steps: ground-plane calibration so distance is in metres rather than pixels, group detection to suppress friends walking together, cross-camera re-identification, a consented staged dataset of following behaviour, and end-to-end latency benchmarks.
