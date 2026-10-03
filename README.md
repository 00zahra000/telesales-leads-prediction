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

1. Reload the CSV with `src.load_data` and then writes it in a table.
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
│   ├── data_analysis.py      # EDA, associations, outliers
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

The app is a batch job and exits after completion. Compose creates the mounted
output folders before the job runs; empty folders alone do not mean it succeeded.
Check the app logs for a failed stage. After changing Python code, use `--build`
to rebuild the image before rerunning the batch.

Loguru writes concise, timestamped stage progress, output summaries and errors
to the console. Detailed tables remain in the Markdown reports. Set
`LOG_LEVEL=DEBUG` in `.env` to include individual training-trial metrics;
the default is `INFO`.

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

Analysis produces dataset and column summaries, target imbalance, purchase-history group summaries, Spearman correlations and categorical associations using Cramér's V. Numeric outliers use 1.5 × IQR fences with investigation notes; values are not removed or clipped. Training uses the fixed feature lists in `src.features`; their selection reasons and timing assumptions are documented below. No separate feature assessment is generated or required.

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

Both models use a fixed random seed for reproducible runs. Training keeps the original class balance and does not adjust predicted probabilities afterward.

The best settings for each model are chosen using validation data, with Average Precision as the main metric. CatBoost wins only if it passes the improvement checks for ranking and probability quality; otherwise, the simpler Logistic Regression model is selected. The selected model stays trained on the training data only.

The saved model is reloaded and checked before evaluation on the separate test data. Evaluation also checks that the database data have not changed since training. Results show how well the model ranks leads, how reliable its probabilities are, and how many purchasers are captured by contacting the highest-scoring 5%, 10%, or 20% of leads. The sales queue is ordered by purchase probability.

See `reports/modeling_report.md` for the results and detailed selection rules from your latest run. Exact selection checks are implemented in `src/train.py`.

## Scoring and outputs

Scoring reads the saved model and metadata, rejects a feature-list mismatch, and reads PostgreSQL inputs without the target. Repeated lead IDs are reduced to the latest `created_at`, with greatest source `id` breaking timestamp ties. Probabilities must be finite and within [0, 1].

`lead_scores` stores lead_id, source_row_id, purchase_probability, priority, score_rank, prediction_timestamp and model_version. Priority 1 is highest; equal probabilities share a dense score rank and lead ID ordering gives unique priorities. The primary key is `(model_version, lead_id)`. Scoring creates the table if necessary and atomically replaces the queue for that model version. Older model versions remain; training creates a new version on each run. Source row IDs describe that ingestion snapshot because subsequent loads reset them.

| Location | Generated content |
| --- | --- |
| `reports/` | Two consolidated Markdown files: `eda_report.md` (dataset, associations, category rates, outliers) and `modeling_report.md` (splits, feature decisions, validation trials, model selection, interpretation, calibration and test metrics) |
| `charts/` | Three EDA charts, validation/test PR, calibration and Top-K charts, plus Logistic Regression coefficient chart |
| `artifacts/` | `model.joblib`, `model_metadata.json`, separate `test_metrics.json` |
| PostgreSQL `lead_scores` | Versioned unique-lead probability queue |

Metadata records model version, UTC training time, features, timing assumptions, split audit, dataset fingerprint, parameters, package versions and validation metrics. Test results and their training-prior reference are stored separately in `artifacts/test_metrics.json`, labeled with the model version, and summarized in `reports/modeling_report.md`. Existing historical reports or feature-ablation artifacts are not recreated by the current pipeline. Output folders are created as needed; reruns overwrite current report/model files without clearing unrelated files.

Informational CSV exports are not generated. Their tables are consolidated into
the two Markdown reports, with feature-pair associations summarized in two sentences. Artifacts
contain the saved model and JSON data needed by training, evaluation and scoring.

To inspect database counts while PostgreSQL is running:

```sh
docker compose up -d db
docker compose exec -T db sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "SELECT COUNT(*) FROM raw_leads;"'
cat reports/modeling_report.md
```

After the batch finishes, run this command from the directory containing
`docker-compose.yml` to view the top 20 leads from the latest scoring run.
Priority **1** is highest; the query filters out older model versions.

```sh
docker compose exec -T db sh -c 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB"' <<'SQL'
SELECT lead_id,
       ROUND(purchase_probability::numeric, 4) AS probability,
       priority
FROM lead_scores
WHERE model_version = (
    SELECT model_version
    FROM lead_scores
    ORDER BY prediction_timestamp DESC
    LIMIT 1
)
ORDER BY priority
LIMIT 20;
SQL
```

## Limitations

- **Ongoing maintenance:** Automatic monitoring and retraining aren’t implemented.
- **Shared model across insurance products:** Third-party and car-body insurance use the same pipeline and model. With more time, separate pipelines could be developed and evaluated for each product to account for differences in customer behavior.
