# Leaving for a Living: Labor Inequality and Financial Mobility of OFWs

**SDGs:** SDG 1 (No Poverty — poverty alleviation, income equality, social protection) and SDG 8 (Decent Work and Economic Growth — job creation, fair wages, economic development).

## Dataset and scope

- Source: PSA Survey on Overseas Filipinos 2024 Public Use File supplied in this repository.
- Records: 3,931; distinct households: 3,403. Household identifiers were used to cluster hypothesis-test standard errors and to keep households together in model evaluation.
- Survey year: 2024. This is not 2025/2026 data. The course guide permits 2024 only if the group documents that it is the latest available; verify and cite that status.
- `RSWGT` is used for descriptive estimates and model fitting/evaluation. Inferential models use unweighted observations with household-cluster robust covariance; the PUF does not provide enough survey-design details for fully design-based standard errors.
- Blanks are missing; zero savings is a valid answer. Savings allocation is reported for 2,655 records. Remittance mode is reported for 2,702; door-to-door is only 3 records.

## RQ1: what drives OFWs toward contractual overseas work

**H0:** Pre-departure domestic employment characteristics and regional demographics have no statistical significance on why OFWs migrate to contractual overseas work.
**H1:** OFWs come disproportionately from lower-wage domestic occupations and from outside the capital region, which gives statistical significance to migrating for contractual labor.

Operationalization: contract-OFW status (`RQ10_REASON = 1`) among people already represented in the overseas-Filipino survey, compared across usual occupation and NCR/outside-NCR household origin. Both are genuinely pre-departure per the questionnaire (C9 occupation "before he/she (last) left"), but the file still does not include non-migrants or domestic wages; occupation is only a prior-work proxy and region is only a rough inequality proxy, so this tests association among surveyed OFWs rather than a true push-factor model.

- Weighted contract-OFW share: **80.3%**.
- Occupation, adjusted for NCR/outside NCR: household-cluster robust linear probability model, Wald F(9)=303.82, p < 0.001, N=3,931; crude Cramer's V=0.336.
- NCR vs outside NCR, adjusted for occupation: household-cluster robust Wald F(1)=8.30, p = 0.004; crude Cramer's V=0.097.

The results support or reject association among surveyed overseas Filipinos; they cannot establish why people migrate or whether lower domestic wages cause contractual migration, since no non-migrant comparison group is available.

## RQ2: demographics and maintaining personal savings abroad

**H0:** An individual's demographic characteristics have no statistically significant relationship with an OFW's allocation of monthly income with regards to personal savings.
**H1:** Having a higher level of educational attainment and a secure contract status make it more likely an OFW maintains personal savings rather than remitting nearly all earnings to their family.

Operationalization: the questionnaire's C18 asks exactly what RQ2 needs (personal savings vs. remittance share of monthly compensation abroad), but `RQ18_*` is absent from the released PUF (the dictionary jumps from C17 to C19). `RQ25_CASHAMT` (C25) also can't substitute — it is a fixed six-month sum (April–Sept 2024) that does not line up with `RQ19_NOMONTH` (months worked over the full five-year window). The closest available proxy is whether any share of what *is* remitted is allocated to savings by the receiving family (`RQ273P_SAVINGS > 0`, `any_savings`, from C27) versus all of it going to consumption/gifts. This is tested against the predictors named in the research question — sex, marital status, destination country (grouped into regions), and contract status — with household-clustered robust standard errors.

- Any-savings share among surveyed OFWs: **47.8%**.
- Sex, adjusted for marital status, destination, and contract: Wald F(1)=43.34, p < 0.001, N=2,655; crude Cramer's V=0.180.
- Marital status, adjusted for the above: Wald F(2)=36.21, p < 0.001; crude Cramer's V=0.153.
- Destination region, adjusted for the above: Wald F(7)=13.46, p = 0.062; crude Cramer's V=0.116.
- Contract status, adjusted for the above: Wald F(1)=0.90, p = 0.342; crude Cramer's V=0.007.

Weighted savings means by marital status:

| Marital status | Weighted mean savings (%) | n |
|---|---:|---:|
| Single | 10.93 | 710 |
| Married | 15.14 | 1675 |
| Previously married | 8.36 | 270 |

Weighted savings means by destination region:

| Destination region | Weighted mean savings (%) | n |
|---|---:|---:|
| Africa | 11.18 | 37 |
| East Asia | 12.58 | 503 |
| Southeast / South Central Asia | 14.15 | 218 |
| Western Asia (Middle East) | 11.36 | 1336 |
| Australia | 18.82 | 90 |
| Europe | 19.20 | 249 |
| Americas | 15.77 | 218 |
| Other | 2.84 | 4 |

## RQ3: remittance transfer mode and regional location

**H0:** The choice of remittance transfer mode is statistically independent of regional location.
**H1:** Recipient households in island and rural areas rely more on OTC or door-to-door transfer compared to urbanized locations.

Operationalization: `RQ26_MODE` is recoded into three channels matching the hypothesis — **Formal Bank**, **Door-to-door**, and **OTC** (agency/local office, friends/co-workers, money-transfer services, and other modes). Geography available is household region (island group) and urban/rural classification, not province.

- Urban/rural terms: household-cluster robust multinomial Wald chi-square(2)=59.16, p < 0.001.
- Island-group terms: household-cluster robust multinomial Wald chi-square(6)=607.59, p < 0.001.
- All location terms jointly: Wald chi-square(8)=1687.00, p < 0.001; crude Cramer's V=0.184 for urban/rural and 0.187 for island group.

Door-to-door has only 3 observations, so any door-to-door-specific conclusion is not reliable; OTC is overwhelmingly driven by formal money-transfer services rather than informal carriers.

## Machine learning (objectives 2 and 3)

**Objective 2 model — predicting any reported savings.** Predictors: demographic traits (age, sex, marital status), human capital (highest grade completed), contract status, and foreign destination (destination region). Household-grouped stratified 80/20 split (seed 132), survey-weighted fitting and scoring. Evaluation sample: 2,655.

| Model | Weighted accuracy | Weighted balanced accuracy | Weighted ROC-AUC |
|---|---:|---:|---:|
| Logistic regression | 0.600 | 0.599 | 0.661 |
| Random forest | 0.606 | 0.605 | 0.646 |
| Weighted majority baseline | 0.522 | 0.500 | 0.500 |

**Objective 3 model — classifying Formal Bank vs. OTC.** Predictors: island group, urban/rural (geography), and monthly income, education, age (socioeconomic status). Door-to-door excluded (3 records, insufficient for train/test splitting). Evaluation sample: 2,699.

| Model | Weighted accuracy | Weighted balanced accuracy | Weighted ROC-AUC |
|---|---:|---:|---:|
| Logistic regression | 0.738 | 0.736 | 0.788 |
| Random forest | 0.712 | 0.712 | 0.772 |
| Weighted majority baseline | 0.560 | 0.500 | 0.500 |

Both models beat their weighted-majority baselines but leave most variance unexplained — treat them as supporting evidence for the determinant groups named in the objectives, not as causal or production-ready classifiers.

## Limitations and next steps

1. The supplied file is from 2024. Document that it is the latest available for this survey if using the course's 2024-data exception.
2. There is no non-migrant comparison group, pre-departure wage field, province identifier, or contract-security measure.
3. The questionnaire's C18 (personal savings vs. remittance allocation of monthly compensation abroad — the field RQ2 names directly) was collected but not released in this PUF; `any_savings` (share of the *remitted* amount the family allocates to savings, from C27) is the closest available substitute, and over one-quarter of records do not report it.
4. OTC is not a single survey category; it is a grouping of agency/local-office, friends/co-workers, money-transfer-service, and other codes, dominated by money-transfer services. Door-to-door is reported only 3 times, so it is excluded from the RQ3 classifier entirely.
5. `RQ24`–`RQ31` (remittance amount, mode, spending shares) are only asked of respondents flagged `RP6M_IND == 1`; this is a questionnaire skip pattern, not random missingness, and it is why RQ2/RQ3 run on roughly 2,600–2,700 records versus 3,931 total.
6. The file has survey weights, but inferential models are not survey-design based because strata/PSU information is not supplied. Cluster-robust standard errors address repeated household records only; interpret p-values cautiously.

## Figures

- `h1_contract_status.png`, `h2_savings_allocation.png`, `h3_mode_by_location.png`
- `rq2_feature_importance.png`, `rq2_confusion_matrix.png`
- `rq3_feature_importance.png`, `rq3_confusion_matrix.png`
- `nutshell.png` / `nutshell.pdf` (1920 x 1080 landscape)
