# Telesales Leads Prediction

Estimate purchase probability for insurance leads and prioritize a batch Telesales queue. The supplied data are synthetic; results do not establish real-world production performance.

## Current workflow

All analysis, training and scoring read PostgreSQL. Only ingestion reads `data/leads.csv`; `data/data_dictionary.csv` is reference metadata and is not loaded into SQL.

```text
CSV → PostgreSQL raw_leads → analysis and feature assessment
                           → chronological split → train-only preprocessing
                           → validation model selection → saved model → test evaluation
                           → batch scoring → PostgreSQL lead_scores
```

1. Reload the CSV with `src.load_data`.
2. Generate EDA and feature assessments with `src.data_analysis`.
3. Train, select, save and evaluate the model with `src.train`.
4. Write the versioned queue with `src.predict`.

Training invokes saved-model evaluation itself. A separate `src.evaluate` run refreshes test reports without fitting or selecting a model again.

## Project layout

Run commands from the directory containing `docker-compose.yml`, `Dockerfile` and `src/`. In this checkout, that is the nested `telesales-leads-prediction/` directory inside the Git repository.

```text
telesales-leads-prediction/
├── data/
│   ├── leads.csv
│   └── data_dictionary.csv
├── sql/init.sql
├── src/
│   ├── db.py                 # database connection from environment variables
│   ├── load_data.py          # transactional CSV reload
│   ├── data_analysis.py      # EDA, associations, outliers and leakage assessment
│   ├── features.py           # shared feature selection and preprocessing
│   ├── train.py              # chronological split, model selection and artifacts
│   ├── evaluate.py           # saved-model test evaluation
│   ├── predict.py            # versioned unique-lead scoring queue
│   └── pipeline.py           # sequential batch runner
├── Dockerfile
├── docker-compose.yml
├── env.example
└── requirements.txt
```

Analysis and training create `reports/`, `charts/` and `artifacts/`. Compose mounts those directories on the host and mounts `data/` read-only.

## Configuration

```sh
cp env.example .env
# Edit .env and choose your local PostgreSQL password.
```

The template supplies `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `DB_HOST` and `DB_PORT`. Compose passes these values to the containers. For Docker, keep `DB_HOST=db` and `DB_PORT=5432`; PostgreSQL has no published host port.

The PostgreSQL password is initialized when the named volume is first created. Changing `.env` later does not update the existing database role password.

## Running with Docker Compose

Requires Docker and Docker Compose v2. The image uses Python 3.12 and the database service uses PostgreSQL 17. Dependencies are installed in a builder-stage virtual environment, then copied into the runtime image. The current Compose configuration uses the default build network.


Run the entire batch:

```sh
docker compose up -d --build
docker compose logs -f app
```

A successful complete batch prints:

```text
Pipeline complete: reports, charts, model and lead scores refreshed.
```

The app is a batch job and exits after completion. Compose creates the mounted
output folders before the job runs; empty folders alone do not mean it succeeded.
Check the app logs for a failed stage. After changing Python code, use `--build`
to rebuild the image before rerunning the batch.

The batch runs integration tests from `tests/`: database connectivity and CSV
availability before ingestion, nonempty loaded data with the source row count
after ingestion, and prediction coverage after scoring. Coverage checks use the
current model version and require exactly one valid probability per unique lead
ID, including repeated source rows through their shared lead ID. A failed check
stops the batch with a nonzero exit status.

To rerun all checks after a completed batch:

```sh
docker compose run --rm app python -m unittest discover -s tests -v
```

`sql/init.sql` creates tables only when PostgreSQL initializes an empty volume. SQL edits do not migrate an existing database. `docker compose down -v` deletes the database volume; use it only for an intentional reset.


## Ingestion and analysis

Each ingestion truncates `raw_leads`, resets surrogate IDs and inserts the CSV in one transaction. A failed insert rolls back. It preserves repeated lead IDs and raw values; empty CSV fields become SQL NULL, literal text such as `NA` stays text, column names become snake_case and creation dates are parsed.

Analysis produces dataset and column summaries, target imbalance, purchase-history group summaries, Spearman correlations and categorical associations using Cramér's V. Numeric outliers use 1.5 × IQR fences with investigation notes; values are not removed or clipped. The scoring-time assessment includes the feature assessment in `reports/eda_report.md` and saves its machine-readable form in `artifacts/feature_assessment.json`. Training requires every selected feature to have an `INCLUDE` recommendation in `artifacts/feature_assessment.json`.

## Features and preprocessing

Training and scoring share 13 explicit inputs in `src.features`:

| Type | Features |
| --- | --- |
| Numeric | minutes_since_abandonment, days_to_policy_expiry, price, discount_percent, sessions_last_7d, offer_views_last_7d |
| Categorical | channel, city, insurance_company, payment_type |
| Binary | has_previous_purchase, visited_offer_page, incoming_call_last_24h |

The code records an experimental assumption that these values were available at abandonment/Telesales eligibility. Historical timing has not been verified. History and event windows must exclude the target purchase and later events, and quote/profile values must represent pre-score snapshots.

The target, identifiers and ingestion timestamp are excluded from predictors. `created_at` controls chronological splitting and scoring snapshot selection. The feature definition excludes device, product_type, partner, price_comparisons_last_7d, days_since_last_visit and expected_margin based on the previously recorded validation reduction experiment. The current pipeline uses this fixed set; it does not rerun feature ablation.

Preprocessing is fitted on training rows only. Numeric inputs use median imputation and, for Logistic Regression, standardization. Binary inputs use most-frequent imputation. Categorical inputs use a constant missing value and one-hot encoding with unknown categories ignored. No additional features are engineered.

## Training and evaluation

Calendar cutoffs are fixed in `src.train`: training precedes July 15, 2026; validation covers July 15 through August 6; test starts August 7. Any lead ID appearing across periods has all its rows purged. Within-period repeats remain. Empty or one-class partitions, incomplete split keys and invalid targets fail validation. Different date coverage may require changing the cutoffs.

Training compares three settings per family:

| Model | Settings |
| --- | --- |
| Logistic Regression | C = 1.0, 0.1, 10.0; maximum 1,000 iterations |
| CatBoost | (iterations, depth, learning rate, L2): (200, 3, 0.05, 5), (300, 2, 0.03, 10), (400, 4, 0.03, 20) |

Both families use seed 42. CatBoost uses one CPU thread. No resampling, class weighting or post-hoc probability calibration is applied.

Each family keeps its best validation Average Precision, breaking ties with Top-10% recall, then Brier score and log loss. CatBoost is selected only when its validation AP improves by at least 0.002, Top-10% recall and lift do not decline, and Brier/log-loss degradation stays within 0.001/0.01. Otherwise Logistic Regression is selected. The winner remains fitted on training data only; there is no train-plus-validation refit.

The selected pipeline is saved and reloaded, and its validation probabilities are checked before test evaluation. Evaluation verifies a fingerprint of the PostgreSQL dataset against training metadata. Metrics include ROC-AUC, Average Precision, log loss, Brier score, calibration diagnostics and precision/recall/lift at Top 5%, 10% and 20%. Capacity counts use `ceil(fraction × rows)`; boundary ties use expected purchasers under random ordering. The fixed 0.5 threshold is diagnostic, while the queue is ordered by probability.

Current results belong in generated reports rather than fixed README metrics: inspect `reports/modeling_report.md` after your run.

## Scoring and outputs

Scoring reads the saved model and metadata, rejects a feature-list mismatch, and reads PostgreSQL inputs without the target. Repeated lead IDs are reduced to the latest `created_at`, with greatest source `id` breaking timestamp ties. Probabilities must be finite and within [0, 1].

`lead_scores` stores lead_id, source_row_id, purchase_probability, priority, score_rank, prediction_timestamp and model_version. Priority 1 is highest; equal probabilities share a dense score rank and lead ID ordering gives unique priorities. The primary key is `(model_version, lead_id)`. Scoring creates the table if necessary and atomically replaces the queue for that model version. Older model versions remain; training creates a new version on each run. Source row IDs describe that ingestion snapshot because subsequent loads reset them.

| Location | Generated content |
| --- | --- |
| `reports/` | Two consolidated Markdown files: `eda_report.md` (dataset, associations, category rates, outliers and leakage assessment) and `modeling_report.md` (splits, feature decisions, validation trials, model selection, interpretation, calibration and test metrics) |
| `charts/` | Three EDA charts, validation/test PR, calibration and Top-K charts, plus Logistic Regression coefficient chart |
| `artifacts/` | `model.joblib`, `model_metadata.json` and machine-readable `feature_assessment.json` used by training |
| PostgreSQL `lead_scores` | Versioned unique-lead probability queue |

Metadata records model version, UTC training time, features, timing assumptions, split audit, dataset fingerprint, parameters, package versions and validation/test metrics. Existing historical reports or feature-ablation artifacts are not recreated by the current pipeline. Output folders are created as needed; reruns overwrite current report/model files without clearing unrelated files.

To inspect database counts while PostgreSQL is running:

```sh
docker compose up -d db
docker compose exec -T db sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "SELECT COUNT(*) FROM raw_leads;"'
cat reports/modeling_report.md
```

Read the version from `artifacts/model_metadata.json`, then query:

```sql
SELECT lead_id, purchase_probability, priority, prediction_timestamp, model_version
FROM lead_scores
WHERE model_version = '<model_version from metadata>'
ORDER BY priority
LIMIT 100;
```

## Limitations

Scores cover the historical dataset, including completed purchases, and demonstrate retrospective batch output. Live contact eligibility, frozen scoring inputs and outcome maturity are not implemented. Synthetic data, temporal prevalence changes and within-period repeated leads limit interpretation. All-data exploration preceded the holdout; validation feature selection is historical, and this is not a fresh prospective evaluation. Purchase propensity does not measure causal contact uplift. Automated monitoring, alerts, feedback and retraining are not implemented.
