# Gradient-Boosting vs MLP Surrogate on Airfoil Self-Noise

This experiment is motivated by "Machine learning aided design of sub-array MIMO antennas for cubesats based on 3D printed metallic ridge gap waveguides" (2022), which uses machine learning to build surrogate models that map antenna geometry to electromagnetic performance, so that design space can be searched without running a full simulation for every candidate. The choice of learning model in that work is not stated, so this experiment asks which standard surrogate family is more accurate on a geometry-to-response problem.

**Question:** Can a cross-validated gradient-boosting surrogate predict a design response from geometry parameters more accurately than a standard MLP surrogate, on a public stand-in dataset?

**Dataset:** UCI Airfoil Self-Noise (1,503 rows, 5 geometry and flow parameters, sound pressure level target), downloaded from the UCI archive. It is a public stand-in for geometry-to-performance modelling. It is not the paper's own data, which is not public.

**Baseline vs improved model:**
- Baseline: scikit-learn `MLPRegressor` with fixed hidden layers and no hyper-parameter search.
- Improved: LightGBM gradient-boosting regressor on the same scaled inputs, with a small grid search chosen by 5-fold cross-validation on the training set only.
- Both use the same 80/20 split (fixed seed), the same input scaling (training statistics), the same evaluation code, and seeds 42, 43 and 44. The main metric is test RMSE in dB. R2 and MAE are secondary metrics.

## How to run

```bash
pip install -r requirements.txt
python analysis.py
```

Results are written to `results.json`. A chart is saved to `figures/rmse_comparison.png`.

## Reference

This project is motivated by the following paper:

> MF Nakmouche, D Deslandes (2022). *Machine learning aided design of sub-array MIMO antennas for cubesats based on 3D printed metallic ridge gap waveguides*. 2022 IEEE Future. https://ieeexplore.ieee.org/abstract/document/10056745/

Paper author: Prof. Ghyslain Gagnon, École de technologie supérieure (ghyslain.gagnon@etsmtl.ca)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
