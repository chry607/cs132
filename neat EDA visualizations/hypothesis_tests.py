"""Hypothesis tests for RQ1-RQ3 on the PSA 2024 Survey on Overseas Filipinos.

Each hypothesis gets two tests:
  * primary  - household-clustered linear probability model; a Wald test asks whether a
               factor's coefficients are all zero while holding the other factors constant
               (chi-square statistic, robust covariance).
               Clustering on RHHNUM accounts for several OFWs from one household.
  * crude    - chi-square test of independence (no controls) with Cramer's V effect size.

H0 is rejected when p < ALPHA. Called from make_all.py.
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2_contingency

import matplotlib.pyplot as plt

from style import ABOVE, INK, MUTED, SAME, finish, legend, new_slide

ALPHA = 0.05

FACTOR_LABELS = {
	"occupation": "Occupation before leaving",
	"capital_origin": "Lived outside NCR",
	"sex": "Sex",
	"marital_status": "Marital status",
	"education": "Education",
	"destination_region": "Destination region",
	"contract_num": "Contract worker",
	"urban_rural": "Urban vs rural",
	"region_group": "Island group",
}
SLIDES = {
	"H1": {
		"file": "rq1_hypothesis_test.png",
		"title": "Prior occupation and living outside NCR are both linked to leaving for contract work",
		"null": "Pre-departure occupation and region have no relationship with leaving for contract work.",
		"sample": "OFWs",
	},
	"H2": {
		"file": "rq2_hypothesis_test.png",
		"title": "Sex and marital status predict saving; education and contract status do not",
		"null": "Demographic traits have no relationship with households putting remittances into savings.",
		"sample": "households that received cash remittances (savings share, C27; C18 not in PUF)",
	},
	"H3": {
		"file": "rq3_hypothesis_test.png",
		"title": "Where a household lives is linked to whether it receives remittances through a bank",
		"null": "Remittance channel (non-bank vs formal bank) is independent of location.",
		"sample": "households that reported a remittance channel (C26); OTC and door-to-door count as non-bank",
	},
}

# Each hypothesis: binary outcome, factors (column -> reference level), alternative hypothesis.
HYPOTHESES = {
	"H1": {
		"question": "Do pre-departure occupation and capital-region origin relate to contract work?",
		"outcome": "contract_num",
		"effect": "on a contract",
		"factors": {"occupation": "Professionals", "capital_origin": "NCR"},
		"alternative": "Lower-wage occupations and origin outside NCR -> more contract work",
	},
	"H2": {
		# C18 (savings kept abroad) is not in the PUF; RQ273P_SAVINGS > 0 is the substitute.
		"question": "Do individual traits relate to allocating any remittance to savings?",
		"outcome": "any_savings_num",
		"effect": "save",
		"factors": {
			"sex": "Male",
			"marital_status": "Single",
			"education": "Below junior high",
			"destination_region": "Western Asia (Middle East)",
			"contract_num": None,
		},
		"alternative": "Higher education and contract status -> more likely to save",
	},
	"H3": {
		"question": "Is remittance channel (non-bank vs formal bank) independent of location?",
		"outcome": "non_bank_num",
		"effect": "use non-bank",
		"factors": {"urban_rural": "Urban", "region_group": "NCR"},
		"alternative": "Rural and island (non-NCR) households -> more OTC / door-to-door",
	},
}


def term(column: str, reference: str | None) -> str:
	return column if reference is None else f"C({column}, Treatment(reference='{reference}'))"


def crude_chi2(sample: pd.DataFrame, factor: str, outcome: str) -> dict:
	table = pd.crosstab(sample[factor], sample[outcome])
	statistic, pvalue, df, expected = chi2_contingency(table)
	n = int(table.to_numpy().sum())
	k = min(table.shape) - 1
	return {
		"statistic": statistic, "df": df, "pvalue": pvalue,
		"cramers_v": np.sqrt(statistic / (n * k)) if k else 0.0,
		"min_expected": expected.min(),
	}


def explain(model, sample, column: str, reference: str | None, effect: str, significant: bool) -> str:
	"""Plain-language result: adjusted rates for the level with the largest gap vs its reference.

	Each rate is the model's average prediction with everyone set to that group and every other
	factor left as observed, so the two rates differ by exactly the tested coefficient."""
	if not significant:
		return "No clear difference once the\nother factors are held constant"
	if reference is None:
		level, reference = 1, 0
	else:
		prefix = f"{term(column, reference)}[T."
		levels = model.params[model.params.index.str.startswith(prefix)]
		top = levels.abs().idxmax()
		level = top[len(prefix):-1]
	rate = lambda value: 100 * model.predict(sample.assign(**{column: value})).mean()
	return f"{level}: {rate(level):.0f}% {effect}\nvs {rate(reference):.0f}% for {reference}"


def run_hypothesis(data: pd.DataFrame, name: str, spec: dict):
	outcome, factors = spec["outcome"], spec["factors"]
	sample = data.dropna(subset=[outcome, *factors]).copy()
	formula = f"{outcome} ~ " + " + ".join(term(c, r) for c, r in factors.items())
	model = smf.ols(formula, data=sample).fit(
		cov_type="cluster", cov_kwds={"groups": pd.factorize(sample["RHHNUM"])[0]}
	)
	wald = model.wald_test_terms(scalar=True).table

	rows = []
	for column, reference in factors.items():
		adjusted = wald.loc[term(column, reference)]
		crude = crude_chi2(sample, column, outcome)
		rows.append({
			"hypothesis": name,
			"factor": column,
			"n": len(sample),
			"wald_stat": float(adjusted["statistic"]),
			"wald_df": int(adjusted["df_constraint"]),
			"wald_p": float(adjusted["pvalue"]),
			"decision": "Reject H0" if adjusted["pvalue"] < ALPHA else "Fail to reject H0",
			"explanation": explain(model, sample, column, reference, spec["effect"], adjusted["pvalue"] < ALPHA),
			"chi2": crude["statistic"],
			"chi2_df": crude["df"],
			"chi2_p": crude["pvalue"],
			"cramers_v": crude["cramers_v"],
			"chi2_min_expected": crude["min_expected"],
		})
	return pd.DataFrame(rows), model


def coefficient_table(model) -> pd.DataFrame:
	"""Percentage-point effect of each level vs its reference, for checking direction."""
	table = pd.DataFrame({
		"effect_pp": 100 * model.params,
		"ci_low_pp": 100 * model.conf_int()[0],
		"ci_high_pp": 100 * model.conf_int()[1],
		"p": model.pvalues,
	}).drop(index="Intercept")
	table.index = table.index.str.replace(r"C\((\w+), Treatment\(reference='(.+)'\)\)\[T\.(.+)\]",
		r"\1: \3 vs \2", regex=True)
	return table


def fmt_p(p: float) -> str:
	return "< 0.001" if p < 0.001 else f"= {p:.3f}"


def card(fig, x, y, w, h, label, p, decision, explanation, significant):
	fill, ink, sub = (ABOVE, "white", "#c9d6f2") if significant else (SAME, INK, "#3d4a63")
	fig.patches.append(plt.Rectangle((x, y), w, h, color=fill, transform=fig.transFigure))
	cx = x + w / 2
	fig.text(cx, y + h * 0.82, label, fontsize=17, weight="bold", color=ink, ha="center", va="center")
	fig.text(cx, y + h * 0.63, ("p < 0.001" if p < 0.001 else f"p = {p:.3f}") + f"  \u00b7  {decision}",
		fontsize=13, color=ink, ha="center", va="center")
	fig.text(cx, y + h * 0.29, explanation, fontsize=10.5, color=sub,
		ha="center", va="center", linespacing=1.3)


def results_slide(path, table, slide):
	fig = new_slide(slide["title"],
		"Adjusted Wald test for each factor, holding the others in the model constant "
		f"(household-clustered); significant at p < {ALPHA}")
	legend(fig, [(ABOVE, "Significant: reject H0"), (SAME, "Not significant: fail to reject H0")])

	w, h, gap = 0.26, 0.22, 0.025
	rows = [table[table["decision"] == "Reject H0"], table[table["decision"] != "Reject H0"]]
	rows = [row for row in rows if len(row)]
	top = 0.45 + (len(rows) * h + (len(rows) - 1) * gap) / 2  # centre the block vertically
	left_edge = 1.0
	for row in rows:
		x = 0.5 - (len(row) * w + (len(row) - 1) * gap) / 2
		left_edge = min(left_edge, x)
		for item in row.itertuples():
			card(fig, x, top - h, w, h, FACTOR_LABELS[item.factor], item.wald_p, item.decision,
				item.explanation, item.decision == "Reject H0")
			x += w + gap
		top -= h + gap

	rejected = [FACTOR_LABELS[f] for f in table.loc[table["decision"] == "Reject H0", "factor"]]
	kept = [FACTOR_LABELS[f] for f in table.loc[table["decision"] != "Reject H0", "factor"]]
	result = "Result: " + "; ".join(filter(None, [
		f"reject H0 for {', '.join(rejected)}" if rejected else "",
		f"fail to reject H0 for {', '.join(kept)}" if kept else "",
	])) + "."
	fig.text(left_edge, top - 0.03, f"H0: {slide['null']}", fontsize=11, color=MUTED, va="top")
	fig.text(left_edge, top - 0.075, result, fontsize=11, weight="bold", color=INK, va="top")
	finish(fig, f"Source: PSA 2024 Survey on Overseas Filipinos (PUF); n = {table['n'].iloc[0]:,} {slide['sample']}.\n"
		"How to read: % = model-estimated share of each group, with the other factors held constant. "
		"Multi-group factors show the group with the largest gap from the comparison group.", path)


def make_hypothesis_tests(data, out):
	results = []
	for name, spec in HYPOTHESES.items():
		table, model = run_hypothesis(data, name, spec)
		results.append(table)

		print(f"\n{'=' * 78}\n{name}: {spec['question']}\nH1 (alternative): {spec['alternative']}")
		print(f"N = {table['n'].iloc[0]:,}  |  alpha = {ALPHA}")
		for row in table.itertuples():
			print(f"  {row.factor:<20} adjusted Wald chi2({row.wald_df}) = {row.wald_stat:7.2f}, "
				f"p {fmt_p(row.wald_p):<8} -> {row.decision}"
				f"   | crude chi2({row.chi2_df}) = {row.chi2:.2f}, V = {row.cramers_v:.3f}")
			if row.chi2_min_expected < 5:
				print(f"  {'':<20} warning: crude chi-square has expected count "
					f"{row.chi2_min_expected:.1f} < 5; rely on the adjusted test")
		print("\n  Direction (percentage-point change vs reference, household-clustered 95% CI):")
		print(coefficient_table(model).round(3).to_string().replace("\n", "\n  "))

	results = pd.concat(results, ignore_index=True)
	results.to_csv(out / "hypothesis_results.csv", index=False)
	for name, table in results.groupby("hypothesis", sort=False):
		results_slide(out / SLIDES[name]["file"], table, SLIDES[name])
	print(f"\nSaved {out / 'hypothesis_results.csv'} and " + ", ".join(s["file"] for s in SLIDES.values()))
