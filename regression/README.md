# Wine Quality Prediction — Regression

This project develops and evaluates regression models for predicting **white wine quality** from physicochemical measurements.

The main objective is not only to obtain a predictive model, but to demonstrate a structured machine-learning experimentation workflow using **MLflow**, controlled model comparisons, Early Stopping, and automated hyperparameter optimization with **Hyperopt**.

## Dataset

The project uses the **White Wine Quality** dataset.

Each observation contains 11 physicochemical features:

- fixed acidity
- volatile acidity
- citric acid
- residual sugar
- chlorides
- free sulfur dioxide
- total sulfur dioxide
- density
- pH
- sulphates
- alcohol

The target variable is the numerical wine quality score.

## Data Split

The dataset is divided into three subsets:

| Split      | Approximate proportion | Purpose                                       |
| ---------- | ---------------------: | --------------------------------------------- |
| Training   |                    60% | Model fitting                                 |
| Validation |                    20% | Model comparison and hyperparameter selection |
| Test       |                    20% | Final unbiased evaluation                     |

The test set is kept completely untouched during model development and is evaluated only after the final configuration has been selected.

## Evaluation Metrics

Three primary regression metrics are used.

### RMSE

Root Mean Squared Error is the primary model-selection metric.

Lower values indicate better performance. RMSE penalizes larger prediction errors more strongly than MAE.

### MAE

Mean Absolute Error measures the average absolute difference between predicted and actual quality scores.

Lower values indicate better performance and the metric is directly interpretable in units of wine quality.

### R²

The coefficient of determination measures how much of the variation in wine quality is explained by the model.

Higher values indicate better performance.

A value near zero corresponds roughly to predicting the mean target value, while negative values indicate performance worse than that baseline.

## Experiment Tracking

MLflow is used to track:

- hyperparameters
- training metrics
- validation metrics
- training time
- prediction time
- epoch-level learning curves
- diagnostic plots
- model signatures
- experiment metadata
- trained model artifacts

The regression experiments are grouped under:

```text
Wine Quality - Regression
```

## Experiment Workflow

The project follows a progressive experimental workflow.

```text
Dummy Baseline
      ↓
Linear Regression
      ↓
Baseline ANN
      ↓
Optimizer Comparison
      ↓
Architecture Comparison
      ↓
Batch Size Comparison
      ↓
Regularization
      ↓
Early Stopping
      ↓
Hyperopt + TPE
      ↓
Winner Verification
      ↓
Final Test Evaluation
      ↓
Model Registry
```

Each manual experiment modifies one primary modeling decision while keeping the remaining configuration fixed. This makes the effect of individual hyperparameters easier to understand.

Hyperopt is introduced only after the controlled experiments establish sensible search ranges.

## Experiment 00 — Dummy Regressor

A `DummyRegressor` predicting the mean target value establishes the minimum baseline.

| Metric | Validation |
| ------ | ---------: |
| RMSE   |     0.8658 |
| MAE    |     0.6602 |
| R²     |    -0.0008 |

The near-zero R² confirms that the model provides essentially no predictive information.

## Experiment 01 — Linear Regression

Linear Regression provides the first meaningful predictive baseline.

| Metric          | Result |
| --------------- | -----: |
| Train RMSE      | 0.7563 |
| Validation RMSE | 0.7412 |
| Validation MAE  | 0.5676 |
| Validation R²   | 0.2665 |

The substantial improvement over the Dummy model confirms that the physicochemical features contain useful predictive information.

However, the remaining error suggests that purely linear relationships are insufficient.

## Experiment 02 — Baseline ANN

The first neural network uses:

```text
11 inputs
   ↓
64 ReLU units
   ↓
1 output
```

Configuration:

```text
Optimizer      SGD
Learning rate  0.01
Momentum       0.0
Batch size     64
Epochs         50
```

Results:

| Metric          | Result |
| --------------- | -----: |
| Train RMSE      | 0.6702 |
| Validation RMSE | 0.6773 |
| Validation MAE  | 0.5202 |
| Validation R²   | 0.3876 |

The nonlinear ANN clearly outperforms Linear Regression.

## Experiment 03 — Optimizer Comparison

The same ANN architecture is trained using different optimizers.

| Optimizer | Train RMSE | Validation RMSE |        MAE |         R² |
| --------- | ---------: | --------------: | ---------: | ---------: |
| SGD       |     0.6702 |      **0.6773** | **0.5202** | **0.3876** |
| Adam      |     0.6935 |          0.6869 |     0.5358 |     0.3702 |
| RMSprop   |     0.6447 |          0.6774 |     0.5323 |     0.3875 |

SGD provides the strongest overall validation performance and is retained for subsequent manual experiments.

## Experiment 04 — Architecture Comparison

Several ANN capacities are evaluated.

| Architecture    | Train RMSE | Validation RMSE |        MAE |         R² |
| --------------- | ---------: | --------------: | ---------: | ---------: |
| 32 → 1          |     0.7035 |          0.6880 |     0.5309 |     0.3681 |
| 64 → 1          |     0.6702 |          0.6773 |     0.5202 |     0.3876 |
| **64 → 32 → 1** | **0.6580** |      **0.6690** | **0.5166** | **0.4026** |
| 128 → 64 → 1    |     0.6551 |          0.6748 |     0.5186 |     0.3920 |

The `64 → 32 → 1` architecture provides the best balance between capacity and generalization.

The larger `128 → 64` network achieves slightly lower training error but worse validation performance.

## Experiment 05 — Batch Size Comparison

The winning architecture is evaluated using different batch sizes.

| Batch Size | Train RMSE | Validation RMSE |        MAE |         R² | Training Time |
| ---------: | ---------: | --------------: | ---------: | ---------: | ------------: |
|         16 |     0.6246 |          0.6947 |     0.5337 |     0.3557 |        7.17 s |
|         32 |     0.6345 |          0.6712 |     0.5206 |     0.3986 |        4.15 s |
|     **64** |     0.6580 |      **0.6690** | **0.5166** | **0.4026** |        2.54 s |
|        128 |     0.6786 |          0.6738 |     0.5212 |     0.3939 |        2.12 s |

Batch size 16 fits the training data most aggressively but generalizes poorly.

Batch size 64 provides the strongest validation performance with reasonable computational cost.

## Experiment 06 — Regularization

L2 regularization and Dropout are compared.

| Configuration  | Train RMSE | Validation RMSE |        MAE |         R² |
| -------------- | ---------: | --------------: | ---------: | ---------: |
| None           |     0.6580 |          0.6690 |     0.5166 |     0.4026 |
| **L2 = 0.001** | **0.6597** |      **0.6674** | **0.5156** | **0.4054** |
| Dropout = 0.2  |     0.7060 |          0.6782 |     0.5236 |     0.3860 |
| L2 + Dropout   |     0.7063 |          0.6777 |     0.5238 |     0.3868 |

A small L2 penalty slightly improves generalization.

Dropout introduces excessive regularization for the relatively small network and decreases performance.

## Experiment 07 — Early Stopping

Instead of fixing training at 50 epochs, training is allowed to continue for up to 200 epochs while monitoring validation RMSE.

Configuration:

```text
Maximum epochs  200
Patience        15
Restore weights True
```

Results:

| Metric          | Result |
| --------------- | -----: |
| Epochs trained  |    121 |
| Best epoch      |    106 |
| Train RMSE      | 0.6278 |
| Validation RMSE | 0.6599 |
| Validation MAE  | 0.5123 |
| Validation R²   | 0.4186 |

The previous fixed limit of 50 epochs was stopping training prematurely.

Early Stopping allows the model to continue learning while preventing unnecessary training once validation performance stops improving.

## Experiment 08 — Hyperparameter Optimization

Hyperopt with the Tree-structured Parzen Estimator algorithm jointly searches:

```text
Architecture
Batch size
Learning rate
Momentum
L2 strength
```

Each Hyperopt trial is tracked as a nested MLflow run.

The best configuration discovered is:

```text
Architecture     128 → 64 → 1
Batch size       32
Learning rate    ≈ 0.001497
Momentum         ≈ 0.2925
L2 strength      ≈ 0.000113
```

Hyperopt results:

| Metric          | Result |
| --------------- | -----: |
| Validation RMSE | 0.6542 |
| Validation MAE  | 0.5086 |
| Validation R²   | 0.4286 |
| Best epoch      |    199 |

The best epoch occurring at 199 indicated that the 200-epoch maximum was constraining training rather than Early Stopping.

## Final Candidate Verification

The Hyperopt winner is therefore retrained with a larger maximum of 400 epochs.

Results:

| Metric          | Result |
| --------------- | -----: |
| Epochs trained  |    232 |
| Best epoch      |    217 |
| Train RMSE      | 0.6075 |
| Validation RMSE | 0.6547 |
| Validation MAE  | 0.5072 |
| Validation R²   | 0.4277 |

Training stops 15 epochs after the best epoch, confirming that Early Stopping rather than the maximum epoch limit determines convergence.

The verification result closely reproduces the best Hyperopt trial.

## Final Test Evaluation

Only after completing model selection is the untouched test set evaluated.

| Metric          | Final Test Result |
| --------------- | ----------------: |
| **RMSE**        |        **0.6783** |
| **MAE**         |        **0.5247** |
| **R²**          |        **0.4059** |
| Prediction time |        0.121599 s |

The test result is slightly weaker than validation performance but remains reasonably consistent with the development results.

The test set is not reused for further tuning.

## Final Model

The selected model uses:

```text
Input features      11

Hidden layers       128 → 64
Activation          ReLU
Output               1

Optimizer           SGD
Learning rate       ≈ 0.001497
Momentum            ≈ 0.2925

L2 strength         ≈ 0.000113
Dropout             0

Batch size          32

Early Stopping
Patience            15
Best epoch          217
Epochs trained      232
```

The exact floating-point values returned by Hyperopt are retained in the notebook and MLflow run rather than relying on their rounded representations above.

## Key Findings

The experiments demonstrate several important modeling and MLOps lessons:

1. Nonlinear ANN models substantially outperform the linear baseline.
2. Lower training error does not necessarily imply better generalization.
3. Increasing model capacity eventually produces diminishing validation improvements.
4. Batch size affects both generalization and computational cost.
5. Mild L2 regularization helps, while Dropout is unnecessary for this model.
6. A fixed epoch count can prematurely terminate learning.
7. Hyperparameters interact, so the best jointly optimized configuration differs from the winners of several one-variable-at-a-time experiments.
8. Hyperopt improves the model by searching combinations rather than optimizing each parameter independently.
9. The validation set is used for model selection while the test set remains untouched until final evaluation.
10. Reproducibility requires tracking model configuration, software dependencies, random seeds, metrics, artifacts, and experiment metadata.

## MLflow Lifecycle

The project uses MLflow for the complete experimental lifecycle:

```text
Experiments
    ↓
Runs
    ↓
Parameters
    ↓
Metrics
    ↓
Artifacts
    ↓
Nested Hyperopt trials
    ↓
Final candidate
    ↓
Test evaluation
    ↓
Logged model
    ↓
Model Registry
```

The final model can be registered as:

```text
WineQualityRegressionANN
```

and a registry alias such as:

```text
champion
```

can be assigned to the selected production candidate.

## Final Performance

The final unbiased performance reported for Project A is:

```text
Test RMSE  0.6783
Test MAE   0.5247
Test R²    0.4059
```

These test metrics represent the final evaluation and are not used for additional model tuning.
