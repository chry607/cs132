import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from data import wmean
from style import ABOVE, BELOW, INK, MUTED, SAME, TRACK, centre_x, finish, legend, move_x, new_slide

SIG_NOTE = "Bar color shows direction vs. the overall average; gray bar fills indicate no clear difference (weighted, household-clustered)."


def rates_by(base, col, min_n=0):
	df = base.dropna(subset=[col])
	rows = {}
	for name, group in df.groupby(col, observed=True):
		test = smf.wls("y ~ grp", df.assign(grp=df[col].eq(name).astype(int)), weights=df["RSWGT"]).fit(
			cov_type="cluster", cov_kwds={"groups": df["RHHNUM"]})
		rows[name] = {"rate": 100 * wmean(group["y"], group["RSWGT"]), "n": len(group), "sig": test.pvalues["grp"] < 0.05}
	out = pd.DataFrame(rows).T.astype({"rate": float, "n": int, "sig": bool}).sort_values("rate")
	return out[out["n"] >= min_n], out[out["n"] < min_n]


def bar_slide(path, groups, title, subtitle, overall, avg_label, legend_items, source, note="", x0=None):
	fig = new_slide(title, subtitle)

	labels, rates, ns, sigs, ys, heads = [], [], [], [], [], []
	pos = 0.0
	for heading, df in groups:
		if heading:
			heads.append((pos, heading))
			pos += 0.8
		for name, row in df.iloc[::-1].iterrows():
			labels.append(name)
			rates.append(row.rate)
			ns.append(row.n)
			sigs.append(row.sig)
			ys.append(pos)
			pos += 1
		pos += 0.5
	total = pos - 0.5

	plot_h = min(0.62, max(0.32, 0.075 * total))
	top = 0.71 if plot_h > 0.5 else 0.66
	ax = fig.add_axes([0.27, top - plot_h, 0.62, plot_h])
	y = np.array(ys)
	show_same = any(color == SAME for color, _ in legend_items)
	colors = [SAME if show_same and not s else ABOVE if r > overall else BELOW for r, s in zip(rates, sigs)]
	used_colors = set(colors)
	legend(fig, [(color, label) for color, label in legend_items if color in used_colors])

	ax.barh(y, 100, color=TRACK, height=0.6, zorder=0)
	ax.axvline(overall, color=INK, lw=1.1, ls=(0, (3, 3)), zorder=0.5, alpha=0.45)
	ax.barh(y, rates, color=colors, height=0.6, zorder=1)
	for yi, rate, c in zip(y, rates, colors):
		if rate > overall:
			ax.vlines(overall, yi - 0.3, yi + 0.3, color="white" if c == ABOVE else INK, lw=1.1,
				ls=(0, (3, 3)), zorder=2, alpha=0.85 if c == ABOVE else 0.45)
	for yi, rate, n in zip(y, rates, ns):
		ax.text(102, yi, f"{rate:.0f}%", va="center", fontsize=13, weight="bold", color=INK)
		ax.text(109, yi, f"n={n:,}", va="center", fontsize=10, color=MUTED)
	for yi, heading in heads:
		ax.text(0, yi - 0.05, heading.upper(), va="center", fontsize=10, weight="bold", color=MUTED)
	ax.annotate(f"{avg_label}: {overall:.0f}%", xy=(overall, 1), xycoords=("data", "axes fraction"),
		xytext=(0, 8), textcoords="offset points", ha="center", va="bottom", fontsize=10.5, color=INK)

	ax.set_yticks(y, labels, fontsize=12.5, color=INK)
	ax.set_xlim(0, 100)
	ax.set_ylim(total - 0.5, -0.6)
	ax.set_xticks([])
	ax.tick_params(length=0, pad=10)
	ax.spines[:].set_visible(False)

	x0 = centre_x(fig, ax) if x0 is None else x0
	move_x(ax, x0)
	if note:
		fig.text(x0, top - plot_h - 0.06, note, fontsize=11, color=MUTED, va="top")
	finish(fig, source, path)
	return x0


def make_h1(data, out):
	base = data.dropna(subset=["RSWGT"]).assign(y=lambda d: d["contract"].astype(float))
	overall = 100 * wmean(base["y"], base["RSWGT"])
	subtitle_1 = "Percentage of OFWs that left for contractual work by prior profession compared to the 80% national average"
	subtitle_2 = "Percentage of OFWs that left for contractual work by region compared to the 80% national average"
	source = ("Source: PSA 2024 Survey on Overseas Filipinos (PUF), n = 3,931. n per bar = unweighted respondents. "
		+ SIG_NOTE.format(who="respondents"))

	occ, _ = rates_by(base, "occupation")
	occ.index = occ.index.str.replace("Non-gainful", "No paid work (non-gainful)")
	region, _ = rates_by(base, "capital_origin")
	gap = region.loc["Outside NCR", "rate"] - region.loc["NCR", "rate"]

	x0 = bar_slide(out / "h1_occupation.png", [("", occ)],
		"OFWs who had elementary occupation are more likely than any other profession to leave because of contractual work",
		subtitle_1, overall, "All overseas Filipinos", [(ABOVE, "Above average"), (BELOW, "Below average")], source)
	bar_slide(out / "h1_region.png", [("", region)],
		"OFWs who lived in NCR are less likely to leave because of contractual work",
		subtitle_2, overall, "All overseas Filipinos", [(ABOVE, "Above average"), (BELOW, "Below average")], source,
		note=f"NCR sits {gap:.0f} percentage points below the rest of the country.", x0=x0)


def make_h2(data, out):
	base = data.dropna(subset=["any_savings_num", "RSWGT"]).assign(y=lambda d: d["any_savings_num"])
	overall = 100 * wmean(base["y"], base["RSWGT"])
	subtitle_1 = "Percentage of remittance-receiving households that put any of the money into savings by gender and relationship status compared to the 48% national average"
	subtitle_2 = "Percentage of remittance-receiving households that put any of the money into savings by country of work of OFW compared to the 48% national average"
	source = (f"Source: PSA 2024 Survey on Overseas Filipinos (PUF); {len(base):,} households that received cash remittances (C27). "
		"n per bar = unweighted. " + SIG_NOTE.format(who="households"))

	sex, _ = rates_by(base, "sex")
	marital, _ = rates_by(base, "marital_status")
	dest, small = rates_by(base, "destination_region", min_n=30)
	dropped = ", ".join(f"{k} (n={v})" for k, v in small["n"].items())

	bar_slide(out / "h2_destination.png", [("", dest)],
		"Households of OFWs in Europe save most often", subtitle_2, overall, "All households", [(ABOVE, "Above average"), (BELOW, "Below average")], source,
		note="Destination differences are not significant once sex, marital status and contract are controlled for (p = 0.06)."
		+ (f" Not shown: {dropped}." if dropped else ""))
	bar_slide(out / "h2_demographics.png", [("Sex of the overseas Filipino", sex), ("Marital status", marital)],
		"Households of male and married OFWs are the most likely to save", subtitle_1, overall, "All households", [(ABOVE, "Above average"), (BELOW, "Below average")], source)


def make_h3(data, out):
	base = data.dropna(subset=["mode", "RSWGT"]).query("mode != 'Door-to-door'")
	base = base.assign(y=base["mode"].eq("OTC").astype(float))
	overall = 100 * wmean(base["y"], base["RSWGT"])
	subtitle = "Share of households receiving remittances through money-transfer services (e.g., Western Union) rather than a bank, survey-weighted"
	source = (f"Source: PSA 2024 Survey on Overseas Filipinos (PUF); {len(base):,} households that reported a remittance channel (C26); door-to-door (n=3) excluded.\n"
		"n per bar = unweighted. " + SIG_NOTE.format(who="households"))

	urban, _ = rates_by(base, "urban_rural")
	island, _ = rates_by(base, "region_group")
	gap = urban.loc["Rural", "rate"] - urban.loc["Urban", "rate"]

	x0 = bar_slide(out / "h3_island_group.png", [("", island)],
		"Mindanao households rely most on money-transfer services", subtitle, overall, "All households", [(ABOVE, "Above average"), (BELOW, "Below average"), (SAME, "No clear difference")], source,
		note="NCR is entirely urban, so the region and urban/rural gaps overlap; both stay significant when modelled together (p < 0.001).")
	bar_slide(out / "h3_urban_rural.png", [("", urban)],
		"Rural households use money-transfer services more than urban ones", subtitle, overall, "All households", [(ABOVE, "Above average"), (BELOW, "Below average")], source,
		note=f"Rural households are {gap:.0f} percentage points more likely to use a money-transfer service.", x0=x0)
