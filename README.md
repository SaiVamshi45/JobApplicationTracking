# JobTrackerAPP

An automated pipeline that reads a Gmail inbox and classifies job-search-related emails — sorting out spam/promotional noise and tagging the real application emails with their intent (rejection, interview invite, next steps, acknowledgment).

## How it works

The pipeline runs in five stages:

1. **Fetch Gmail** — Authenticates with the Gmail API (OAuth2) and pulls emails matching a date query (e.g. `after:2025/8/11`).
2. **Preprocess** — Strips HTML, normalizes umlauts (Deutsche→ASCII), removes URLs/tracking artifacts, and concatenates subject + body into a single text field.
3. **Stage 1 — Category classification**: labels each email as one of
   - `job_related`
   - `non_job`
   - `promotional`
   - `job_agency_spam`
4. **Stage 1.5 — Authenticity filter**: rule-based pass (keyword + known recruiter/ATS sender lists — Workday, Greenhouse, Lever, SuccessFactors, etc.) to catch false positives/negatives from the ML stage before the more expensive Stage 2 model runs.
5. **Stage 2 — Intent classification**: for emails that passed the filter, predicts the specific intent —
   - `rejection`
   - `interview`
   - `next_step`
   - `acknowledgment`

Final output is a table of `Sender | Subject | predicted_intent` for the job-related emails in the queried window.

## Models

Both stages were prototyped with two architectures, trained and compared side by side (confusion matrix, per-class precision/recall/F1, ROC-AUC):

| Stage | Model | Notes |
|---|---|---|
| 1 (category) | CNN (Keras, tokenizer + embedding) | Baseline |
| 1 (category) | XLM-RoBERTa (fine-tuned, HF `Trainer`) | Multilingual (EN/DE) |
| 2 (intent) | CNN (Keras) | Baseline |
| 2 (intent) | XLM-RoBERTa (fine-tuned) | Used for runtime inference (`./xlmr_stage2`) |

## Repo structure

```
JobTrackerAPP/
├── codefilestoupload/
│   ├── JobTrackerApplicationUntilStage2_vivek.ipynb   # full pipeline: training + inference
│   └── credentials.json                               # Gmail OAuth client secret (DO NOT COMMIT)
├── datasets/
│   ├── Stage1_ideal_balanced_final.csv                # 5,956 rows — category-labeled training data
│   └── stage2_final_clean.csv                         # 400 rows — intent-labeled training data
├── Resultswithownmail/                                 # Stage 1/1.5/2 outputs from a run on one mailbox
└── ResultwithVivekemail/                                # Stage 1/1.5/2 outputs from a run on another mailbox
```

### Datasets

- **`Stage1_ideal_balanced_final.csv`** (5,956 rows) — columns: `Sender, Subject, preprocessed_text, category`. Roughly balanced across the four Stage-1 categories (1,300–1,625 examples each).
- **`stage2_final_clean.csv`** (400 rows) — columns: `Sender, Subject, preprocessed_text, category, language, intent_label`. English/German mix (217/183); intents skew toward `rejection` (165) and `next_step` (94).

### Results folders

Each `Results*` folder holds the CSV output of one pipeline run over a mailbox, one file per stage:
`stage1_results_*.csv` → `stage1_5_filtered_*.csv` / `stage1_5_passed_*.csv` → `stage2_results_*.csv` / `stage2_predictions_*.csv`.

## Setup

```bash
pip install transformers datasets evaluate accelerate
pip install tensorflow torch joblib google-api-python-client google-auth google-auth-oauthlib langdetect beautifulsoup4
```

### Gmail API access

1. Create an OAuth 2.0 client (Desktop app) in Google Cloud Console with the Gmail API enabled.
2. Download the client secret and save it as `credentials.json` in the working directory the notebook runs from.
3. On first run, `gmail_authenticate()` / `get_credentials()` will open a browser consent flow and cache the resulting token as `token.pickle` for subsequent runs.
4. Scope used: `gmail.readonly`.

**Note:** `credentials.json` and `token.pickle` are per-user secrets and should not be committed to version control — add them to `.gitignore`. The copy currently in `codefilestoupload/` should be treated as sensitive.

## Usage

Open `codefilestoupload/JobTrackerApplicationUntilStage2_vivek.ipynb` (built for Google Colab):

1. Run the setup/import cells.
2. (Optional) Re-run the Stage 1 / Stage 2 training cells if retraining on new data — trained artifacts (label encoders, tokenizer, model weights) are saved under `stage1_models/` and `xlmr_stage2/`.
3. Set `DATE_QUERY` (e.g. `"after:2025/8/11"`) and run the fetch → preprocess → Stage 1 → Stage 1.5 → Stage 2 inference cells.
4. Inspect/export results — final predictions are saved to CSV with `Sender, Subject, predicted_intent`.

## Status

Prototype / research notebook stage — training, EDA, and inference all currently live in one notebook. Not yet refactored into a standalone script or service.
