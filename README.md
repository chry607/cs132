# Leaving for a Living: Measuring Labor Inequality and Inaccessibility through the Impact of Macroeconomic and Regional Factors on the Financial Mobility of OFWs

CS 132 project analyzing the **PSA Survey on Overseas Filipinos (SOF) 2024 Public Use File** to study why Filipinos migrate for contractual work, what predicts whether they keep personal savings, and how their remittance channel choice relates to geography.

**SDGs addressed**
- **SDG 1 — No Poverty:** poverty alleviation, income equality, social protection
- **SDG 8 — Decent Work and Economic Growth:** job creation, fair wages, economic development

## Research questions and hypotheses

**RQ1 — What drives OFWs toward contractual overseas work, given local employment conditions?**
- H0: Pre-departure domestic employment characteristics and regional demographics have no statistical significance on why OFWs migrate to contractual overseas work.
- H1: OFWs come disproportionately from lower-wage domestic occupations and from outside the capital region.

**RQ2 — What individual characteristics predict whether an OFW maintains personal savings abroad vs. remitting nearly 100% of earnings home?**
- H0: An individual's demographic characteristics have no statistically significant relationship with an OFW's allocation of monthly income with regards to personal savings.
- H1: Higher educational attainment and a secure contract status make it more likely an OFW maintains personal savings.

**RQ3 — Does an OFW's regional location predict their remittance transfer mode (Formal Bank vs. Door-to-door vs. OTC)?**
- H0: The choice of remittance transfer mode is statistically independent of regional location.
- H1: Recipient households in island and rural areas rely more on OTC or door-to-door transfer than urbanized locations.

**Objectives**
1. Identify the factors that push Filipinos to work abroad vs. finding work domestically.
2. Model the determinants of savings/remittance allocation using demographic traits, human capital, contract status, and foreign destination.
3. Classify the primary remittance channel of OFWs using geographical location and socioeconomic status.

## Data

- **Source:** `PHL-PSA-SOF-2024-PUF 2/` — PSA Survey on Overseas Filipinos 2024 Public Use File (3,931 records, 3,403 households).
- `SOF PUF 2024.CSV` — the microdata.
- `SOF 2024 Questionnaire.pdf` — the source questionnaire (SOF Form 1); used to verify every variable's exact wording and skip pattern before analysis.
- `sof_2024_metadata(dictionary).xlsx` — variable dictionary and value-set codebook.

**Important data-quality finding:** RQ2 cites questionnaire item **C18** (how an OFW allocates monthly compensation earned abroad between remittance, consumption, personal savings, and other — i.e., the literal "personal savings abroad" measure). Cross-checking the questionnaire against the dictionary confirms C18 was asked **but was not included in this public release** (the variable list jumps directly from C17 to C19). The analysis substitutes `RQ273P_SAVINGS` (C27 — the share of money *already remitted* that the receiving family allocates to savings) as the closest available proxy, and this substitution is documented throughout the notebook and report. See `outputs/eda_report.md` for the full set of such data-fidelity notes (skip patterns, sentinel-value checks, pre-departure variable confirmation).

## Project structure

```
main.ipynb              Full analysis notebook — run top to bottom to reproduce everything below
requirements.txt        Python dependencies
outputs/                Generated on run (gitignored): report, figures, CSVs
PHL-PSA-SOF-2024-PUF 2/ Raw PUF data, questionnaire, and codebook
```

## Running it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook main.ipynb   # run all cells
```

This regenerates everything in `outputs/`: `eda_report.md`, all figures, `hypothesis_tests.csv`, and `model_metrics_savings.csv` / `model_metrics_mode.csv`.

## Methodology

- **Hypothesis tests:** linear-probability / multinomial-logit models with **household-clustered robust standard errors** (so repeated household records don't understate uncertainty), plus crude chi-square / Cramér's V for unadjusted association.
- **Survey weights (`RSWGT`)** are used for all descriptive statistics and ML model fitting/evaluation.
- **Two machine-learning classifiers** (logistic regression + random forest, evaluated against a weighted-majority baseline, with a household-grouped stratified 80/20 split so no household leaks across train/test):
  - **Objective 2 model** — predicts whether an OFW reports any savings, from demographic traits, human capital, contract status, and foreign destination.
  - **Objective 3 model** — classifies remittance channel (Formal Bank vs. OTC) from geography and socioeconomic status.

## Results

### RQ1 — Contract status and pre-departure characteristics

- Weighted contract-OFW share: **80.3%**.
- Occupation (pre-departure, domestic), adjusted for NCR origin: Wald F(9) = 303.82, **p < 0.001**, Cramér's V = 0.34.
- NCR vs. outside-NCR origin, adjusted for occupation: Wald F(1) = 8.30, **p = 0.004**, Cramér's V = 0.10.
- **H0 rejected.** Both prior occupation and regional origin are significantly associated with contract status, consistent with H1 — but this is association among people who already migrated; there is no non-migrant comparison group, so a causal "push factor" claim is not supported.

### RQ2 — Demographics and savings allocation

- Any-savings share: **47.8%**.
- Sex: Wald F(1) = 43.34, **p < 0.001**. Marital status: Wald F(2) = 36.21, **p < 0.001**.
- Destination region: Wald F(7) = 13.46, p = 0.062 (not significant at 0.05).
- **Contract status: Wald F(1) = 0.90, p = 0.342 — not significant.**
- Married OFWs save the most (15.1% weighted mean), previously-married the least (8.4%); Europe/Australia destinations save more (~19%) than the Middle East (~11%).
- **Partial support for H1.** Demographics (sex, marital status) matter; contract status — the mechanism H1 specifically names — does not. The result should be read as testing the C27 proxy, not the C18 construct the hypothesis literally asks about.

### RQ3 — Remittance mode and location

- Urban/rural terms: Wald chi-square(2) = 59.16, **p < 0.001**. Island-group terms: Wald chi-square(6) = 607.59, **p < 0.001**.
- Rural households rely on OTC far more than urban households; Mindanao is the most OTC-dependent region, NCR the least.
- **H0 clearly rejected — strongly supports H1.** Caveat: door-to-door has only 3 observations in the entire file, so it cannot be evaluated on its own; "OTC" here is effectively dominated by formal money-transfer services (Western Union, MoneyGram, etc.), not informal carriers.

### Machine learning

| Model | Target | Weighted ROC-AUC | vs. baseline |
|---|---|---:|---|
| Savings classifier (obj. 2) | Any savings? | 0.66 | Modest lift |
| Channel classifier (obj. 3) | Formal Bank vs. OTC | **0.79** | Clear lift |

- Savings model: age, education, and sex are the strongest predictors; **contract status has ~zero feature importance**, matching the non-significant hypothesis test.
- Channel model: **income and education matter more than raw geography** — financial-channel access looks more like a socioeconomic-status story than a pure location story once income is controlled for.

### Bottom line

- **Strongest result:** RQ3 — remittance channel choice is clearly geography-driven.
- **Most caveated result:** RQ1 — real associations, but no non-migrant baseline to support a causal push-factor claim.
- **Most surprising/counter-hypothesis result:** contract status does not predict savings behavior in either the statistical test or the ML model, contradicting RQ2's stated H1.
- **Biggest data-fidelity issue:** the exact variable RQ2 names (C18) is not in the released PUF — confirmed by checking the source questionnaire, not just inferred from missing data.

Full statistics, all data-dictionary cross-checks, and every figure are in `outputs/eda_report.md` after running the notebook.

## Limitations

1. The file is 2024 data only — document this is the latest available if relying on the course's 2024-data exception.
2. No non-migrant comparison group, pre-departure wage field, province identifier, or contract-security measure exists in the PUF.
3. C18 (personal savings abroad) was asked but not released; `any_savings` (from C27) is a proxy, not the literal construct.
4. OTC is a grouping of several non-bank, non-door-to-door codes, dominated by money-transfer services; door-to-door (n=3) is excluded from the ML classifier entirely.
5. `RQ24`–`RQ31` (remittance amount, mode, spending) follow a questionnaire skip pattern (`RP6M_IND == 1` only) — not random missingness — which is why RQ2/RQ3 run on ~2,600–2,700 records vs. 3,931 total.
6. Survey weights are used for fitting/scoring, but inferential models are not fully survey-design-based (no strata/PSU in the PUF); cluster-robust SEs address repeated household records only.

## Data citation

Philippine Statistics Authority (PSA). *Survey on Overseas Filipinos (SOF) 2024, Public Use File.*
