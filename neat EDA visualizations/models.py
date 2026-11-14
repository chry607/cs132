import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data import wmean
from style import ABOVE, BELOW, INK, MUTED, SAME, centre_x, finish, legend, move_x, new_slide

def fit_savings_model(data: pd.DataFrame):
	sample = data.dropna(subset=["savings_pct", "RSWGT"]).copy()
	sample = sample.loc[sample["RSWGT"].gt(0)]
	numeric = ["RQ4_AGE", "RQ8_HGRADE"]
	categorical = ["sex", "marital_status", "contract", "destination_region"]
	X = sample[numeric + categorical].copy()
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
	assert set(sample.iloc[train_index]["RHHNUM"]).isdisjoint(sample.iloc[test_index]["RHHNUM"])
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
	pipelines: dict[str, Pipeline] = {}
	predictions: dict[str, np.ndarray] = {}
	for name, estimator in estimators.items():
		model = Pipeline([("preprocess", transform), ("model", estimator)])
		model.fit(X_train, y_train, model__sample_weight=w_train / w_train.mean())
		probabilities = model.predict_proba(X_test)[:, 1]
		prediction = (probabilities >= 0.5).astype(int)
		results[name] = {
			"n_train": len(y_train), "n_test": len(y_test),
			"weighted_accuracy": accuracy_score(y_test, prediction, sample_weight=w_test),
			"weighted_balanced_accuracy": balanced_accuracy_score(y_test, prediction, sample_weight=w_test),
			"weighted_roc_auc": roc_auc_score(y_test, probabilities, sample_weight=w_test),
			"f1_macro": f1_score(y_test, prediction, average="macro"),
		}
		pipelines[name] = model
		predictions[name] = prediction
	prevalence = wmean(y_train.astype(float), w_train)
	results["Weighted majority baseline"] = {
		"n_train": len(y_train), "n_test": len(y_test),
		"weighted_accuracy": max(prevalence, 1 - prevalence),
		"weighted_balanced_accuracy": 0.5, "weighted_roc_auc": 0.5, "f1_macro": np.nan,
	}
	return results, len(sample), pipelines, (y_test, predictions, X_test, w_test)


def fit_mode_model(data: pd.DataFrame):
	sample = data.dropna(subset=["mode", "RSWGT"]).copy()
	sample = sample.loc[sample["mode"].isin(["Formal Bank", "OTC"])]
	sample = sample.loc[sample["RSWGT"].gt(0)]
	numeric = ["RQ4_AGE", "RQ17_AVEMINC"]
	categorical = ["region_group", "urban_rural", "education"]
	X = sample[numeric + categorical].copy()
	for column in categorical:
		X[column] = X[column].astype("object")
	y = sample["mode"].eq("OTC").astype(int)
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
	assert set(sample.iloc[train_index]["RHHNUM"]).isdisjoint(sample.iloc[test_index]["RHHNUM"])
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
	pipelines: dict[str, Pipeline] = {}
	predictions: dict[str, np.ndarray] = {}
	for name, estimator in estimators.items():
		model = Pipeline([("preprocess", transform), ("model", estimator)])
		model.fit(X_train, y_train, model__sample_weight=w_train / w_train.mean())
		probabilities = model.predict_proba(X_test)[:, 1]
		prediction = (probabilities >= 0.5).astype(int)
		results[name] = {
			"n_train": len(y_train), "n_test": len(y_test),
			"weighted_accuracy": accuracy_score(y_test, prediction, sample_weight=w_test),
			"weighted_balanced_accuracy": balanced_accuracy_score(y_test, prediction, sample_weight=w_test),
			"weighted_roc_auc": roc_auc_score(y_test, probabilities, sample_weight=w_test),
			"f1_macro": f1_score(y_test, prediction, average="macro"),
		}
		pipelines[name] = model
		predictions[name] = prediction
	prevalence = wmean(y_train.astype(float), w_train)
	results["Weighted majority baseline"] = {
		"n_train": len(y_train), "n_test": len(y_test),
		"weighted_accuracy": max(prevalence, 1 - prevalence),
		"weighted_balanced_accuracy": 0.5, "weighted_roc_auc": 0.5, "f1_macro": np.nan,
	}
	return results, len(sample), pipelines, (y_test, predictions, X_test, w_test)

def confusion_slide(path, matrix, labels, title, subtitle, stats, source):
	ramp = mcolors.LinearSegmentedColormap.from_list("blue", ["#f2f5fa", ABOVE])
	m = np.asarray(matrix, dtype=float)
	share = m / m.sum(axis=1, keepdims=True)
	fig = new_slide(title, subtitle)

	ax = fig.add_axes([0.20, 0.16, 0.34, 0.58])
	for i in range(2):
		for j in range(2):
			c = ramp(share[i, j])
			ax.add_patch(plt.Rectangle((j + 0.02, i + 0.02), 0.96, 0.96, color=c))
			tc = "white" if 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2] < 0.5 else INK
			ax.text(j + 0.5, i + 0.44, f"{share[i, j]:.0%}", ha="center", va="center", fontsize=30, weight="bold", color=tc)
			ax.text(j + 0.5, i + 0.68, f"{int(m[i, j]):,} households", ha="center", va="center", fontsize=11.5, color=tc)
	ax.set_xlim(0, 2)
	ax.set_ylim(2, 0)
	ax.set_xticks([0.5, 1.5], [f"Predicted:\n{l}" for l in labels], fontsize=12, color=INK)
	ax.xaxis.tick_top()
	ax.set_yticks([0.5, 1.5], [f"Actually:\n{l}\n(n={int(m[i].sum())})" for i, l in enumerate(labels)], fontsize=12, color=INK)
	ax.tick_params(length=0, pad=10)
	ax.spines[:].set_visible(False)
	ax.text(1, 2.08, "Row percentages: each row adds up to 100%. Darker = larger share.", ha="center", va="top", fontsize=10, color=MUTED)

	y = 0.70
	for big, small in stats:
		fig.text(0.62, y, big, fontsize=30, weight="bold", color=ABOVE, va="top")
		fig.text(0.62, y - 0.072, small, fontsize=12.5, color=INK, va="top", linespacing=1.2)
		y -= 0.19
	finish(fig, source, path)


def importance_slide(path, imp, group, colors, title, subtitle, note, source):
	imp = imp.clip(lower=0).sort_values(ascending=False)
	fig = new_slide(title, subtitle)
	shown = {group[k] for k in imp.index[imp > 0]}
	legend(fig, [(c, label) for label, c in colors.items() if label in shown])

	ax = fig.add_axes([0.30, 0.31, 0.55, 0.42])
	y = np.arange(len(imp))
	top = imp.max()
	ax.barh(y, imp.values, color=[colors[group[k]] for k in imp.index], height=0.6)
	for yi, v in zip(y, imp.values):
		if v >= 0.005:
			ax.text(v + top * 0.02, yi, f"{v:.2f}", va="center", fontsize=13, weight="bold", color=INK)
		elif v > 0:
			ax.text(v + top * 0.02, yi, "<0.01  almost none", va="center", fontsize=12, color=MUTED)
		else:
			ax.text(top * 0.02, yi, "0.00  no measurable effect", va="center", fontsize=12, color=MUTED)
	ax.set_yticks(y, imp.index, fontsize=12.5, color=INK)
	ax.set_ylim(len(imp) - 0.4, -0.6)
	ax.set_xlim(0, top * 1.25)
	ax.set_xticks([])
	ax.tick_params(length=0, pad=10)
	ax.spines[:].set_visible(False)

	x0 = centre_x(fig, ax)
	move_x(ax, x0)
	fig.text(x0, 0.235, note, fontsize=11, color=MUTED, va="top", linespacing=1.2)
	finish(fig, source, path)


def importance(pipeline, X_test, y_test, w_test, names):
	result = permutation_importance(pipeline, X_test, y_test, scoring="roc_auc", n_repeats=50,
		random_state=132, sample_weight=w_test.to_numpy())
	return pd.Series(result.importances_mean, index=[names[c] for c in X_test.columns])


def model_stats(cm, majority_row, majority_label):
	acc = np.trace(cm) / cm.sum()
	guess = cm[majority_row].sum() / cm.sum()
	return f"{acc:.0%} correct", f"vs. {guess:.0%} by always guessing \"{majority_label}\""


def make_rq2(data, out):
	results, n, pipes, (y_test, preds, X_test, w_test) = fit_savings_model(data)
	cm = confusion_matrix(y_test, preds["Random forest"], labels=[0, 1])
	rf_auc, lr_auc = results["Random forest"]["weighted_roc_auc"], results["Logistic regression"]["weighted_roc_auc"]
	split = f"{n:,} remittance-receiving households (C27), 80/20 split by household"

	confusion_slide(out / "rq2_confusion_matrix.png", cm, ["No savings", "Any savings"],
		"The savings model does only slightly better than guessing",
		f"Random forest predictions for {len(y_test)} held-out households that were not used to train the model",
		[model_stats(cm, 0, "no savings"),
		 (f"{cm[1, 1] / cm[1].sum():.0%} of savers found", f"{cm[1, 1]} of the {cm[1].sum()} households that saved\nwere predicted to save"),
		 (f"AUC {rf_auc:.2f}", "0.5 = coin flip, 1.0 = perfect (survey-weighted).\nSex, marital status, age and education\ngive only a weak signal")],
		f"Source: PSA 2024 Survey on Overseas Filipinos (PUF); {split} so no household is in both sets.\n"
		"Outcome = household put any of the remittance into savings. Inputs: age, education, sex, marital status, contract, destination. "
		"Percentages are unweighted; AUC is survey-weighted.")

	names = {"sex": "Sex", "marital_status": "Marital status", "RQ4_AGE": "Age", "RQ8_HGRADE": "Education",
		"contract": "Contract status", "destination_region": "Destination region"}
	group = {"Sex": "Personal background", "Marital status": "Personal background", "Age": "Personal background",
		"Education": "Named in H1 (education, contract)", "Contract status": "Named in H1 (education, contract)",
		"Destination region": "Destination"}
	colors = {"Personal background": ABOVE, "Named in H1 (education, contract)": BELOW, "Destination": SAME}
	importance_slide(out / "rq2_feature_importance.png",
		importance(pipes["Random forest"], X_test, y_test, w_test, names), group, colors,
		"Sex and marital status matter most; education adds nothing",
		"How much the model's accuracy (AUC) drops when each input about the OFW is scrambled. Bigger drop = more useful to the model.",
		f"Random forest, {len(y_test)} held-out households, 50 shuffles each, survey-weighted.\n"
		"The logistic regression ranks the same top three and also finds no effect of education.\n"
		"Matches the H2 tests: sex and marital status are significant, contract status is not.",
		f"Source: PSA 2024 Survey on Overseas Filipinos (PUF); {split}.\n"
		f"Target = household put any of the remittance into savings. Model AUC = {rf_auc:.2f} (random forest), {lr_auc:.2f} (logistic regression).")


def make_rq3(data, out):
	results, n, pipes, (y_test, preds, X_test, w_test) = fit_mode_model(data)
	cm = confusion_matrix(y_test, preds["Random forest"], labels=[0, 1])
	rf_auc, lr_auc = results["Random forest"]["weighted_roc_auc"], results["Logistic regression"]["weighted_roc_auc"]
	split = f"{n:,} households receiving through a bank or money-transfer service (C26), 80/20 split by household"

	confusion_slide(out / "rq3_confusion_matrix.png", cm, ["Bank", "Money-transfer"],
		"The channel model gets about 7 in 10 households right",
		f"Random forest predictions for {len(y_test)} held-out households that were not used to train the model",
		[model_stats(cm, 1, "money-transfer"),
		 (f"{cm[0, 0] / cm[0].sum():.0%} and {cm[1, 1] / cm[1].sum():.0%}", "of bank users and money-transfer users\nrespectively were identified correctly"),
		 (f"AUC {rf_auc:.2f}", "0.5 = coin flip, 1.0 = perfect (survey-weighted).\nIncome and education carry most of the signal")],
		f"Source: PSA 2024 Survey on Overseas Filipinos (PUF); {split}.\n"
		"Inputs: OFW's income, education and age; household's island group and urban/rural. Door-to-door (n=3) excluded. "
		"Percentages are unweighted; AUC is survey-weighted.")

	names = {"RQ17_AVEMINC": "OFW's monthly income", "education": "OFW's education", "region_group": "Island group (incl. NCR)",
		"urban_rural": "Urban or rural household", "RQ4_AGE": "OFW's age"}
	group = {"OFW's monthly income": "OFW's income and education", "OFW's education": "OFW's income and education",
		"Island group (incl. NCR)": "Household location", "Urban or rural household": "Household location", "OFW's age": "Other"}
	colors = {"OFW's income and education": ABOVE, "Household location": BELOW, "Other": SAME}
	importance_slide(out / "rq3_feature_importance.png",
		importance(pipes["Random forest"], X_test, y_test, w_test, names), group, colors,
		"Income and education predict the remittance channel better than location",
		"How much the model's accuracy (AUC) drops when each input is scrambled. Bigger drop = more useful to the model.",
		f"Random forest, {len(y_test)} held-out households, 50 shuffles each, survey-weighted. The logistic regression gives the same top three\n"
		"(education, income, island group). Location still adds information, just less than income and education.",
		f"Source: PSA 2024 Survey on Overseas Filipinos (PUF); {split}.\n"
		f"Target = money-transfer service vs. bank. Model AUC = {rf_auc:.2f} (random forest), {lr_auc:.2f} (logistic regression).")
