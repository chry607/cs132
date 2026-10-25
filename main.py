from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from patsy import dmatrix
from scipy.stats import chi2_contingency
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.discrete.discrete_model import MNLogit
import statsmodels.formula.api as smf


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "PHL-PSA-SOF-2024-PUF 2" / "SOF PUF 2024.CSV"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

OCCUPATION_RANGES = {
	"Armed forces": range(1, 4),
	"Managers": range(11, 15),
	"Professionals": range(21, 27),
	"Technicians": range(31, 36),
	"Clerical support": range(41, 45),
	"Services and sales": range(51, 55),
	"Agriculture, forestry, fishery": range(61, 64),
	"Craft and trades": range(71, 76),
	"Plant and machine operators": range(81, 84),
	"Elementary occupations": range(91, 97),
	"Non-gainful": [9],
}
OCCUPATION_LABELS = {
	code: label for label, codes in OCCUPATION_RANGES.items() for code in codes
}
EDU_GROUPS = {
	0: "Below junior high", 1: "Below junior high", 2: "Below junior high",
	3: "Below junior high", 4: "High school", 5: "High school",
	6: "High school", 7: "Post-secondary / TVET", 8: "Post-secondary / TVET",
	9: "College or higher", 10: "College or higher", 11: "College or higher",
}
EDU_ORDER = ["Below junior high", "High school", "Post-secondary / TVET", "College or higher"]


def region_group(code: float) -> str | float:
	if pd.isna(code):
		return np.nan
	code = int(code)
	if code == 13:
		return "NCR"
	if code in {14, 1, 2, 3, 4, 17, 5}:
		return "Luzon"
	if code in {6, 7, 8}:
		return "Visayas"
	if code in {9, 10, 11, 12, 16, 19}:
		return "Mindanao"
	return np.nan


def load_data() -> pd.DataFrame:
	data = pd.read_csv(DATA_PATH, dtype=str, encoding="utf-8-sig")
	numeric = [
		"RQ3_SEX", "RQ4_AGE", "RQ8_HGRADE", "RQ9_2DGT", "RQ10_REASON",
		"RQ17_AVEMINC", "RQ26_MODE", "RQ273P_SAVINGS", "RREG", "RURB2020", "RSWGT",
	]
	for column in numeric:
		data[column] = pd.to_numeric(data[column].str.strip(), errors="coerce")

	data["contract"] = data["RQ10_REASON"].eq(1)
	data["contract_num"] = data["contract"].astype(int)
	data["occupation"] = data["RQ9_2DGT"].map(
		lambda value: OCCUPATION_LABELS.get(int(value)) if pd.notna(value) else np.nan
	)
	data["education"] = data["RQ8_HGRADE"].map(EDU_GROUPS)
	data["region_group"] = data["RREG"].map(region_group)
	data["capital_origin"] = data["region_group"].map(
		lambda value: "NCR" if value == "NCR" else "Outside NCR" if pd.notna(value) else np.nan
	)
	data["urban_rural"] = data["RURB2020"].map({1: "Urban", 2: "Rural"})
	data["sex"] = data["RQ3_SEX"].map({1: "Male", 2: "Female"})
	data["savings_pct"] = data["RQ273P_SAVINGS"].where(data["RQ273P_SAVINGS"].between(0, 100))
	data["any_savings"] = data["savings_pct"].gt(0)
	data["mode"] = data["RQ26_MODE"].map({1: "Bank", 5: "Money transfer service"})
	data.loc[data["RQ26_MODE"].notna() & data["mode"].isna(), "mode"] = "Other reported mode"
	data["mode_code"] = data["RQ26_MODE"].map({1: 0, 5: 1})
	data.loc[data["RQ26_MODE"].notna() & data["mode_code"].isna(), "mode_code"] = 2
	return data


def wmean(values: pd.Series, weights: pd.Series) -> float:
	valid = values.notna() & weights.notna() & weights.gt(0)
	return float(np.average(values[valid], weights=weights[valid])) if valid.any() else np.nan


def chi_result(table: pd.DataFrame) -> dict[str, float | int]:
	statistic, pvalue, dof, expected = chi2_contingency(table)
	dimension = min(table.shape[0] - 1, table.shape[1] - 1)
	n = int(table.to_numpy().sum())
	return {
		"n": n, "chi2": float(statistic), "df": int(dof), "p": float(pvalue),
		"cramers_v": float(np.sqrt(statistic / (n * dimension))) if dimension else 0.0,
		"min_expected": float(expected.min()),
	}


def clustered_ols_terms(formula: str, data: pd.DataFrame) -> tuple[object, pd.DataFrame]:
	fitted = smf.ols(formula, data=data).fit(
		cov_type="cluster", cov_kwds={"groups": data["RHHNUM"]}
	)
	return fitted, fitted.wald_test_terms(scalar=True).table


def term_result(terms: pd.DataFrame, term: str) -> dict[str, float | int]:
	row = terms.loc[term]
	return {
		"statistic": float(row["statistic"]),
		"pvalue": float(row["pvalue"]),
		"df": int(row["df_constraint"]),
	}


def clustered_multinomial_tests(data: pd.DataFrame) -> tuple[dict[str, dict[str, float | int]], object]:
	design = dmatrix("1 + C(urban_rural) + C(region_group)", data, return_type="dataframe")
	fitted = MNLogit(data["mode_code"].astype(int), design).fit(
		disp=False,
		maxiter=200,
		cov_type="cluster",
		cov_kwds={"groups": data["RHHNUM"]},
	)

	def joint_test(term: str) -> dict[str, float | int]:
		rows = [index for index, name in enumerate(design.columns) if term in name]
		n_equations = fitted.params.shape[1]
		indices = [
			np.ravel_multi_index((row, equation), fitted.params.shape, order="F")
			for row in rows
			for equation in range(n_equations)
		]
		restriction = np.eye(fitted.params.size)[indices]
		test = fitted.wald_test(restriction, scalar=True)
		return {
			"statistic": float(test.statistic),
			"pvalue": float(test.pvalue),
			"df": len(indices),
		}

	results = {
		"Urban/rural location": joint_test("C(urban_rural)"),
		"Island group": joint_test("C(region_group)"),
		"All location terms": joint_test("C("),
	}
	return results, fitted


def pvalue(value: float) -> str:
	return "< 0.001" if value < 0.001 else f"= {value:.3f}"


def plot_h1(data: pd.DataFrame) -> None:
	fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
	valid = data.dropna(subset=["occupation", "RSWGT"])
	occupation_rates = valid.groupby("occupation", observed=True).apply(
		lambda group: 100 * wmean(group["contract"].astype(float), group["RSWGT"]),
		include_groups=False,
	).sort_values()
	axes[0].barh(occupation_rates.index, occupation_rates.values, color="#176b63")
	axes[0].set(title="Contract OFW share by prior occupation", xlabel="Weighted share (%)", xlim=(0, 100))
	capital = data.dropna(subset=["region_group", "RSWGT"]).copy()
	capital["origin"] = np.where(capital["region_group"].eq("NCR"), "NCR", "Outside NCR")
	rates = capital.groupby("origin").apply(
		lambda group: 100 * wmean(group["contract"].astype(float), group["RSWGT"]),
		include_groups=False,
	).reindex(["NCR", "Outside NCR"])
	axes[1].bar(rates.index, rates.values, color="#dd8a35")
	axes[1].set(title="Contract OFW share by household region", ylabel="Weighted share (%)", ylim=(0, 100))
	fig.suptitle("Contract status and observed origins", fontsize=16, weight="bold")
	fig.tight_layout()
	fig.savefig(OUT / "h1_contract_status.png", dpi=180, bbox_inches="tight")
	plt.close(fig)


def plot_h2(data: pd.DataFrame) -> pd.DataFrame:
	valid = data.dropna(subset=["savings_pct", "education", "RSWGT"]).copy()
	valid["education"] = pd.Categorical(valid["education"], EDU_ORDER, ordered=True)
	education = valid.groupby("education", observed=True).apply(
		lambda group: pd.Series({
			"weighted_mean_pct": wmean(group["savings_pct"], group["RSWGT"]),
			"n": len(group),
		}), include_groups=False,
	).reset_index()
	contract = valid.groupby("contract", observed=True).apply(
		lambda group: pd.Series({
			"weighted_mean_pct": wmean(group["savings_pct"], group["RSWGT"]),
			"n": len(group),
		}), include_groups=False,
	).reset_index()
	fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
	axes[0].bar(education["education"], education["weighted_mean_pct"], color="#3278a3")
	axes[0].set(title="Savings share by education", xlabel="Highest grade completed", ylabel="Weighted mean savings (%)")
	axes[0].tick_params(axis="x", rotation=18)
	axes[1].bar(contract["contract"].map({False: "No contract", True: "Contract"}), contract["weighted_mean_pct"], color="#c85442")
	axes[1].set(title="Savings share by contract status", ylabel="Weighted mean savings (%)")
	fig.suptitle("Reported allocation of remittances to savings", fontsize=16, weight="bold")
	fig.tight_layout()
	fig.savefig(OUT / "h2_savings_allocation.png", dpi=180, bbox_inches="tight")
	plt.close(fig)
	return education


def plot_h3(data: pd.DataFrame) -> None:
	valid = data.dropna(subset=["mode", "urban_rural", "region_group"])
	fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
	for axis, location, title in [
		(axes[0], "urban_rural", "Mode by urban/rural household"),
		(axes[1], "region_group", "Mode by island group"),
	]:
		share = pd.crosstab(valid[location], valid["mode"], normalize="index").mul(100)
		share = share.reindex(
			index=["Urban", "Rural"] if location == "urban_rural" else ["Luzon", "Visayas", "Mindanao", "NCR"],
			columns=["Bank", "Money transfer service", "Other reported mode"],
			fill_value=0,
		)
		share.plot(
			kind="bar", stacked=True, ax=axis, color=["#176b63", "#dd8a35", "#6c7780"]
		)
		axis.set(title=title, ylabel="Respondents (%)", xlabel="")
		axis.tick_params(axis="x", rotation=15)
		axis.legend().remove()
	handles, labels = axes[0].get_legend_handles_labels()
	fig.legend(handles, labels, title="Reported mode", loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.02), frameon=False)
	fig.suptitle("Remittance channel and household location", fontsize=16, weight="bold")
	fig.tight_layout(rect=[0, 0.09, 1, 0.94])
	fig.savefig(OUT / "h3_mode_by_location.png", dpi=180, bbox_inches="tight")
	plt.close(fig)


def fit_models(data: pd.DataFrame) -> tuple[dict[str, dict[str, float]], int]:
	sample = data.dropna(subset=["savings_pct", "RSWGT"]).copy()
	sample = sample.loc[sample["RSWGT"].gt(0)]
	features = ["RQ4_AGE", "RQ8_HGRADE", "RQ17_AVEMINC", "sex", "occupation", "contract", "region_group", "urban_rural"]
	numeric = ["RQ4_AGE", "RQ8_HGRADE", "RQ17_AVEMINC"]
	categorical = ["sex", "occupation", "contract", "region_group", "urban_rural"]
	X = sample[features].copy()
	for column in categorical:
		X[column] = X[column].astype("object")
	y = sample["savings_pct"].gt(0).astype(int)
	weights = sample["RSWGT"].astype(float)
	transform = ColumnTransformer([
		("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
		("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical),
	])
	train_index, test_index = next(
		StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=132).split(
			X, y, groups=sample["RHHNUM"]
		)
	)
	train_households = set(sample.iloc[train_index]["RHHNUM"])
	test_households = set(sample.iloc[test_index]["RHHNUM"])
	assert train_households.isdisjoint(test_households)
	X_train, X_test = X.iloc[train_index], X.iloc[test_index]
	y_train, y_test = y.iloc[train_index], y.iloc[test_index]
	w_train, w_test = weights.iloc[train_index], weights.iloc[test_index]
	estimators = {
		"Logistic regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
		"Random forest": RandomForestClassifier(
			n_estimators=300, min_samples_leaf=8, class_weight="balanced_subsample",
			random_state=132, n_jobs=-1,
		),
	}
	results: dict[str, dict[str, float]] = {}
	for name, estimator in estimators.items():
		model = Pipeline([("preprocess", transform), ("model", estimator)])
		model.fit(X_train, y_train, model__sample_weight=w_train / w_train.mean())
		probabilities = model.predict_proba(X_test)[:, 1]
		prediction = probabilities >= 0.5
		results[name] = {
			"n_train": len(y_train), "n_test": len(y_test),
			"weighted_accuracy": accuracy_score(y_test, prediction, sample_weight=w_test),
			"weighted_balanced_accuracy": balanced_accuracy_score(y_test, prediction, sample_weight=w_test),
			"weighted_roc_auc": roc_auc_score(y_test, probabilities, sample_weight=w_test),
		}
	prevalence = wmean(y_train.astype(float), w_train)
	results["Weighted majority baseline"] = {
		"n_train": len(y_train), "n_test": len(y_test),
		"weighted_accuracy": max(prevalence, 1 - prevalence),
		"weighted_balanced_accuracy": 0.5, "weighted_roc_auc": 0.5,
	}
	return results, len(sample)


def plot_nutshell(data: pd.DataFrame, models: dict[str, dict[str, float]]) -> None:
	fig, axes = plt.subplots(1, 3, figsize=(13.34, 7.5), dpi=144)
	fig.patch.set_facecolor("#f3f0e8")
	fig.suptitle("Overseas work, savings, and remittances", fontsize=22, weight="bold", y=0.975)
	fig.text(0.5, 0.91, "A snapshot from the PSA Survey on Overseas Filipinos 2024", ha="center", fontsize=12)

	regional = data.dropna(subset=["region_group", "RSWGT"]).groupby("region_group", observed=True).apply(
		lambda group: 100 * wmean(group["contract"].astype(float), group["RSWGT"]),
		include_groups=False,
	).sort_values()
	axes[0].barh(regional.index, regional.values, color="#176b63")
	axes[0].set(title="Contract OFW share", xlabel="Weighted share (%)", xlim=(0, 100))

	savings = data.dropna(subset=["education", "savings_pct", "RSWGT"])
	education = savings.groupby("education", observed=True).apply(
		lambda group: wmean(group["savings_pct"], group["RSWGT"]), include_groups=False
	).reindex(EDU_ORDER)
	axes[1].bar(range(len(education)), education.values, color="#3278a3")
	axes[1].set_xticks(range(4), ["< JH", "HS", "Post-sec.", "College+"])
	axes[1].set(title="Remittance savings share", ylabel="Weighted mean (%)")

	modes = data.dropna(subset=["mode", "urban_rural"])
	table = pd.crosstab(modes["urban_rural"], modes["mode"], normalize="index").mul(100)
	table.reindex(index=["Urban", "Rural"], columns=["Bank", "Money transfer service", "Other reported mode"], fill_value=0).plot(
		kind="bar", stacked=True, ax=axes[2], color=["#176b63", "#dd8a35", "#6c7780"], legend=False
	)
	axes[2].set(title="Remittance channel", ylabel="Respondents (%)", xlabel="")
	axes[2].tick_params(axis="x", rotation=0)
	axes[2].legend().remove()
	handles, labels = axes[2].get_legend_handles_labels()
	fig.legend(handles, labels, title="Mode", loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.19), frameon=False, fontsize=8)

	auc = models["Random forest"]["weighted_roc_auc"]
	fig.text(0.5, 0.145, "Occupation and origin track contract classification; education tracks savings allocation, but contract status does not. Channels vary by location.", ha="center", fontsize=8)
	fig.text(0.5, 0.10, f"Savings model held-out weighted ROC-AUC: {auc:.2f} | Association is not causation", ha="center", fontsize=10, weight="bold")
	fig.text(0.5, 0.055, "Household region / urban-rural are geography proxies; province and explicit OTC categories are unavailable.", ha="center", fontsize=8)
	for axis in axes:
		axis.spines[["top", "right"]].set_visible(False)
		axis.grid(axis="y", alpha=0.2)
		axis.set_axisbelow(True)
	fig.tight_layout(rect=[0.03, 0.22, 0.97, 0.87])
	fig.savefig(OUT / "nutshell.png", dpi=144, facecolor=fig.get_facecolor())
	fig.savefig(OUT / "nutshell.pdf", facecolor=fig.get_facecolor())
	plt.close(fig)


def main() -> None:
	plt.style.use("seaborn-v0_8-whitegrid")
	data = load_data()
	h1_data = data.dropna(subset=["occupation", "capital_origin"]).copy()
	h1_model, h1_terms = clustered_ols_terms(
		"contract_num ~ C(occupation) + C(capital_origin)", h1_data
	)
	h1_occupation = term_result(h1_terms, "C(occupation)")
	h1_location = term_result(h1_terms, "C(capital_origin)")
	h1_crude_occupation = chi_result(pd.crosstab(h1_data["occupation"], h1_data["contract"]))
	h1_crude_location = chi_result(pd.crosstab(h1_data["capital_origin"], h1_data["contract"]))

	savings = data.dropna(subset=["savings_pct", "education", "contract_num"]).copy()
	h2_model, h2_terms = clustered_ols_terms(
		"savings_pct ~ C(education) + contract_num", savings
	)
	h2_education = term_result(h2_terms, "C(education)")
	h2_contract = term_result(h2_terms, "contract_num")
	modes = data.dropna(subset=["mode", "mode_code", "region_group", "urban_rural"]).copy()
	h3_tests, h3_model = clustered_multinomial_tests(modes)
	h3_urban = chi_result(pd.crosstab(modes["urban_rural"], modes["mode"]))
	h3_region = chi_result(pd.crosstab(modes["region_group"], modes["mode"]))

	models, model_n = fit_models(data)
	plot_h1(data)
	education_summary = plot_h2(data)
	plot_h3(data)
	plot_nutshell(data, models)

	reported_savings = int(data["savings_pct"].notna().sum())
	reported_modes = int(data["mode"].notna().sum())
	door_count = int(data["RQ26_MODE"].eq(4).sum())
	contract_share = 100 * wmean(data["contract"].astype(float), data["RSWGT"])
	savings_mean = wmean(data["savings_pct"], data["RSWGT"])
	education_text = "| Education group | Weighted mean savings (%) | n |\n|---|---:|---:|\n"
	education_text += "\n".join(
		f"| {row.education} | {row.weighted_mean_pct:.2f} | {int(row.n)} |"
		for row in education_summary.itertuples(index=False)
	)
	model_text = "\n".join(
		f"| {name} | {m['weighted_accuracy']:.3f} | {m['weighted_balanced_accuracy']:.3f} | {m['weighted_roc_auc']:.3f} |"
		for name, m in models.items()
	)
	report = f"""# Exploratory Data Analysis and Modeling

## Dataset and scope

- Source: PSA Survey on Overseas Filipinos 2024 Public Use File supplied in this repository.
- Records: {len(data):,}; distinct households: {data['RHHNUM'].nunique():,}. Household identifiers were used to cluster hypothesis-test standard errors and to keep households together in model evaluation.
- Survey year: {int(data['RSVYYR'].dropna().mode().iloc[0])}. This is not 2025/2026 data. The course guide permits 2024 only if the group documents that it is the latest available; verify and cite that status.
- `RSWGT` is used for descriptive estimates and model fitting/evaluation. Inferential models use unweighted observations with household-cluster robust covariance; the PUF does not provide enough survey-design details for fully design-based standard errors.
- Blanks are missing; zero savings is a valid answer. Savings allocation is reported for {reported_savings:,} records. Remittance mode is reported for {reported_modes:,}; door-to-door is only {door_count} records.

## RQ1: contract status and observed origins

Operationalization: contract-OFW status (`RQ10_REASON = 1`) among people already represented in the overseas-Filipino survey, compared across usual occupation and NCR/outside-NCR household origin. The file does not include non-migrants or domestic wages; occupation is only a prior-work proxy and region is only a rough inequality proxy.

- Weighted contract-OFW share: **{contract_share:.1f}%**.
- Occupation, adjusted for NCR/outside NCR: household-cluster robust linear probability model, Wald F({h1_occupation['df']})={h1_occupation['statistic']:.2f}, p {pvalue(h1_occupation['pvalue'])}, N={len(h1_data):,}; crude Cramer's V={h1_crude_occupation['cramers_v']:.3f}.
- NCR vs outside NCR, adjusted for occupation: household-cluster robust Wald F({h1_location['df']})={h1_location['statistic']:.2f}, p {pvalue(h1_location['pvalue'])}; crude Cramer's V={h1_crude_location['cramers_v']:.3f}.

The results support association among surveyed overseas Filipinos, not why people migrate or whether lower domestic wages cause contractual migration.

## RQ2: reported savings allocation

Outcome: percentage of remittances allocated to savings (`RQ273P_SAVINGS`), not personal savings from all monthly income. Blank allocations are excluded. The model uses education groups and contract status and clusters uncertainty by household.

- Weighted mean savings allocation among reported values: **{savings_mean:.1f}%**.
- Education group, adjusted for contract status: clustered linear model Wald F({h2_education['df']})={h2_education['statistic']:.2f}, p {pvalue(h2_education['pvalue'])}, N={len(savings):,}.
- Contract status, adjusted for education group: clustered linear model Wald F({h2_contract['df']})={h2_contract['statistic']:.2f}, p {pvalue(h2_contract['pvalue'])}.

Weighted savings means by education group:

{education_text}

This is partial evidence for an education association, not for the proposed contract-security effect. The survey records OFW-with-contract status, not security or quality of contract.

## RQ3: remittance mode and location

Available geography is household region and urban/rural classification, not province. Rare channel categories are grouped as “Other reported mode”; OTC is not a separate survey category. The clustered multinomial model tests the joint location terms across Bank, Money transfer service, and Other reported mode.

- Urban/rural terms: household-cluster robust multinomial Wald chi-square({h3_tests['Urban/rural location']['df']})={h3_tests['Urban/rural location']['statistic']:.2f}, p {pvalue(h3_tests['Urban/rural location']['pvalue'])}.
- Island-group terms: household-cluster robust multinomial Wald chi-square({h3_tests['Island group']['df']})={h3_tests['Island group']['statistic']:.2f}, p {pvalue(h3_tests['Island group']['pvalue'])}.
- All location terms jointly: Wald chi-square({h3_tests['All location terms']['df']})={h3_tests['All location terms']['statistic']:.2f}, p {pvalue(h3_tests['All location terms']['pvalue'])}; crude Cramer's V={h3_urban['cramers_v']:.3f} for urban/rural and {h3_region['cramers_v']:.3f} for island group.

The available data support location-channel associations, but cannot verify province-level effects or the specific door-to-door/OTC alternative. Door-to-door has only {door_count} observations.

## Machine learning: any reported savings

Target: whether savings allocation is greater than zero. Predictors: age, education, overseas monthly income, sex, prior occupation, contract-OFW status, household region group, and urban/rural classification. Imputation and encoding are fitted on training data only. A stratified five-fold household-group split provides an approximately 80/20 holdout (seed 132); no household appears in both partitions. Survey weights are used for model fitting and holdout metrics.

Evaluation sample: {model_n:,} respondents with reported savings allocation.

| Model | Weighted accuracy | Weighted balanced accuracy | Weighted ROC-AUC |
|---|---:|---:|---:|
{model_text}

Compare the classifiers to the weighted majority baseline. Scores indicate modest predictive signal and do not establish effects or causation.

## Nutshell plot

Headline: **Overseas Work, Savings, and Remittances**

Caption: Contract classification is associated with prior occupation and household origin among surveyed overseas Filipinos. Education group is associated with the share of remittances assigned to savings, but contract status is not. Remittance channel also varies across urban/rural and broad island group. These findings do not establish why people migrate: this 2024 file excludes non-migrants and lacks pre-departure wages, contract security, province, and explicit OTC coding; door-to-door appears only three times.

## Limitations and next steps

1. The supplied file is from 2024. Document that it is the latest available for this survey if using the course's 2024-data exception.
2. There is no non-migrant comparison group, pre-departure wage field, province identifier, explicit OTC category, or contract-security measure.
3. The savings measure is allocation of remittances, and over one-quarter of records do not report it; missingness can bias estimates.
4. The file has survey weights, but inferential models are not survey-design based because strata/PSU information is not supplied. Cluster-robust standard errors address repeated household records only; interpret p-values cautiously.

For direct tests of the stated alternatives, supplement this PUF with non-migrant employment/wage records and data with province, explicit transfer channel, contract security, and survey-design documentation.

## Figures

- `h1_contract_status.png`
- `h2_savings_allocation.png`
- `h3_mode_by_location.png`
- `nutshell.png` and `nutshell.pdf` (1920 x 1080 landscape)
"""
	(OUT / "eda_report.md").write_text(report, encoding="utf-8")

	tests = [
		{"test": "H1 occupation adjusted for origin", "n": len(h1_data), **h1_occupation},
		{"test": "H1 NCR origin adjusted for occupation", "n": len(h1_data), **h1_location},
		{"test": "H2 education group adjusted for contract", "n": len(savings), **h2_education},
		{"test": "H2 contract adjusted for education", "n": len(savings), **h2_contract},
		{"test": "H3 urban-rural location terms", "n": len(modes), **h3_tests["Urban/rural location"]},
		{"test": "H3 island group terms", "n": len(modes), **h3_tests["Island group"]},
		{"test": "H3 all location terms jointly", "n": len(modes), **h3_tests["All location terms"]},
	]
	pd.DataFrame(tests).to_csv(OUT / "hypothesis_tests.csv", index=False)
	pd.DataFrame(models).T.to_csv(OUT / "model_metrics.csv", index_label="model")
	print(f"Analysis complete: {OUT}")
	print(f"Records: {len(data):,}; households: {data['RHHNUM'].nunique():,}")
	print(f"Savings reported: {reported_savings:,}; modes reported: {reported_modes:,}")


if __name__ == "__main__":
	main()
