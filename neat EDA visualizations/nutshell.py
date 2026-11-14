import base64
import io
import json
import math
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors, font_manager
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
from shapely.geometry import shape
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

from data import DATA_PATH, ROOT

THEMES = {
	"deck": dict(
		background="white", text="#0f1e3d", muted="#5b6880", divider="#e3e8f0",
		map_colors=["#dbe5f5", "#93b4e8", "#5b8ad8", "#1e3a8a"], map_side="#c3cee0", outline="white", accent="#1e3a8a",
		font=["Avenir Next", "Helvetica Neue", "Roboto"], title=dict(y=103, size=48, va="baseline"),
		subtitle=dict(y=145, size=26, va="baseline"), name="nutshell"),
	"course": dict(
		background="#1B181C", text="#FFFFFF", muted="#C8C1CB", divider="#3A333D",
		map_colors=["#4B2362", "#9E3E7F", "#DE6B61", "#EDB081"], map_side="#49304F", outline="#D9C8DD", accent="#FFB000",
		font=["Roboto"], title=dict(y=52, size=44, va="top"), subtitle=dict(y=122, size=24, va="top"), name="nutshell_course"),
}

TITLE = "Most OFW families in Mindanao receive money through transfer services"
SUBTITLE = "Contract work, savings and remittance channels among overseas Filipinos, 2024"
MAP_LEGEND = "Map: share of families receiving remittances through money-transfer services instead of banks"
CREDIT = "Data: PSA Survey on Overseas Filipinos 2024 PUF. Chart: CS 132 project team. Boundaries: J. Faeldon (2023 vintage, MIT)."
LABELS = {
	13: ("NCR", -320, -10), 14: ("Cordillera", -320, -40), 7: ("Central Visayas", 230, -30),
	5: ("Bicol", 250, -50), 16: ("Caraga", 200, -20), 19: ("BARMM", -330, 10), 12: ("SOCCSKSARGEN", 210, 45),
}
MINDANAO = {9, 10, 11, 12, 16, 19}
DEPTH, TILT = 15, 0.86
MAP_TOP, MAP_H, MAP_CX = 262, 700, 640


def load_assets():
	assets = ROOT / ".plain_map_assets"
	assets.mkdir(exist_ok=True)
	nb = json.loads((ROOT / "nutshell_plain.ipynb").read_text(encoding="utf-8"))
	with zipfile.ZipFile(io.BytesIO(base64.b64decode(nb["metadata"]["map_assets"]))) as archive:
		for member in archive.infolist():
			if not (assets / member.filename).resolve().is_relative_to(assets.resolve()):
				raise ValueError("Invalid asset path")
		archive.extractall(assets)
	for font in (assets / "assets/fonts").glob("*.ttf"):
		font_manager.fontManager.addfont(str(font))
	return json.loads((assets / "assets/geography/regions_2023.geojson").read_text())


def summarise():
	df = pd.read_csv(DATA_PATH, dtype=str, encoding="utf-8-sig")
	for c in ["RREG", "RSWGT", "RQ10_REASON", "RQ273P_SAVINGS", "RQ26_MODE", "RQ3_SEX"]:
		df[c] = pd.to_numeric(df[c].str.strip(), errors="coerce")
	df = df[df.RSWGT.gt(0)]
	df["contract"] = df.RQ10_REASON.eq(1).astype(float)
	savings = df.RQ273P_SAVINGS.where(df.RQ273P_SAVINGS.between(0, 100))
	df["any_sav"] = savings.gt(0).astype(float).where(savings.notna())
	df["mt"] = df.RQ26_MODE.map({1: 0.0, 2: 1.0, 3: 1.0, 5: 1.0, 6: 1.0})

	def share(d, col):
		d = d[d[col].notna()]
		return 100 * np.average(d[col], weights=d.RSWGT)

	region_mt = {int(r): share(g, "mt") for r, g in df.groupby("RREG")}
	region_n = {int(r): int(g.mt.notna().sum()) for r, g in df.groupby("RREG")}
	stats = {
		"contract": share(df, "contract"),
		"contract_ncr": share(df[df.RREG.eq(13)], "contract"),
		"contract_out": share(df[df.RREG.ne(13)], "contract"),
		"save": share(df, "any_sav"),
		"save_m": share(df[df.RQ3_SEX.eq(1)], "any_sav"),
		"save_f": share(df[df.RQ3_SEX.eq(2)], "any_sav"),
		"mt": share(df, "mt"),
		"mt_min": share(df[df.RREG.isin(MINDANAO)], "mt"),
		"mt_ncr": share(df[df.RREG.eq(13)], "mt"),
	}
	return region_mt, region_n, stats


def make_nutshell(out, theme="deck"):
	t = THEMES[theme]
	geometry = load_assets()
	region_mt, region_n, s = summarise()

	geoms = {r: unary_union([shape(f["geometry"]) for f in geometry["features"] if f["properties"]["region"] == r]) for r in region_mt}
	geoms = {r: g if g.is_valid else g.buffer(0) for r, g in geoms.items()}
	polys = lambda g: [g] if g.geom_type == "Polygon" else list(g.geoms)

	def merc(p):
		lon, lat = p[:2]
		return np.array([math.radians(lon), -math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))])

	pts = np.array([merc(p) for g in geoms.values() for pg in polys(g) for p in pg.exterior.coords])
	low, high = pts.min(0), pts.max(0)
	scale = MAP_H / (TILT * (high[1] - low[1]))
	x_off = MAP_CX - scale * (high[0] - low[0]) / 2
	project = lambda p: np.array([x_off + (merc(p) - low)[0] * scale, MAP_TOP + (merc(p) - low)[1] * scale * TILT])

	def path(g, depth=0):
		verts, codes = [], []
		for pg in polys(g):
			pg = orient(pg, sign=1.0)
			for ring in [pg.exterior, *pg.interiors]:
				q = [project(p) + [0, depth] for p in ring.coords]
				verts += q
				codes += [MplPath.MOVETO] + [MplPath.LINETO] * (len(q) - 2) + [MplPath.CLOSEPOLY]
		return MplPath(np.asarray(verts), codes)

	cmap = colors.LinearSegmentedColormap.from_list("mt", t["map_colors"])
	norm = colors.Normalize(vmin=30, vmax=80, clip=True)
	plt.rcParams.update({"font.family": t["font"], "pdf.fonttype": 42, "svg.fonttype": "none"})
	fig = plt.figure(figsize=(6.4, 3.6), dpi=300, facecolor=t["background"])
	ax = fig.add_axes([0, 0, 1, 1])
	ax.set_xlim(0, 1920)
	ax.set_ylim(1080, 0)
	ax.axis("off")

	def label(x, y, text, size=24, weight="normal", ha="left", color=t["text"], va="top"):
		return ax.text(x, y, text, fontsize=size * 72 / 300, color=color, fontweight=weight, ha=ha, va=va, linespacing=1.15)

	label(80, t["title"]["y"], TITLE, size=t["title"]["size"], weight="bold", va=t["title"]["va"])
	label(80, t["subtitle"]["y"], SUBTITLE, size=t["subtitle"]["size"], color=t["muted"], va=t["subtitle"]["va"])

	for depth in range(DEPTH, 0, -3):
		for g in geoms.values():
			ax.add_patch(PathPatch(path(g, depth), facecolor=t["map_side"], edgecolor=t["map_side"], linewidth=0.15, zorder=1))
	for r, g in geoms.items():
		ax.add_patch(PathPatch(path(g), facecolor=cmap(norm(region_mt[r])), edgecolor=t["outline"], linewidth=0.25, zorder=2))

	centers = {r: project([g.centroid.x, g.centroid.y]) for r, g in geoms.items()}
	for r, (name, dx, dy) in LABELS.items():
		x, y = centers[r]
		tx, ty = x + dx, y + dy
		ax.plot([x, tx + (0 if dx > 0 else 150)], [y, ty + 14], color=t["muted"], linewidth=0.3, zorder=5)
		label(tx, ty, f"{name} {region_mt[r]:.0f}%", size=23)

	label(80, 186, MAP_LEGEND, size=21, weight="bold")
	ax.imshow(np.linspace(30, 80, 400).reshape(1, -1), extent=(80, 480, 242, 222), cmap=cmap, norm=norm, aspect="auto")
	for x, value, ha in [(94, "30%", "left"), (280, "55%", "center"), (466, "80%", "right")]:
		label(x, 248, value, size=18, color=t["muted"], ha=ha)

	ax.plot([1230, 1230], [205, 930], color=t["divider"], linewidth=0.6)
	blocks = [
		("RQ1: CONTRACT WORK", f"{s['contract']:.0f}%", "of overseas Filipinos left as contract workers.\n"
			f"The share is {s['contract_out']:.0f}% outside NCR and {s['contract_ncr']:.0f}% in NCR,\nand 95–97% for elementary, craft and machine workers."),
		("RQ2: SAVINGS", f"{s['save']:.0f}%", "of families saved part of the remittance they received.\n"
			f"The share is {s['save_m']:.0f}% for families of male OFWs and {s['save_f']:.0f}%\nfor female OFWs. Contract status had no effect."),
		("RQ3: REMITTANCE CHANNEL", f"{s['mt']:.0f}%", "of families receive remittances through money-transfer\n"
			f"services. The share is {s['mt_min']:.0f}% in Mindanao and {s['mt_ncr']:.0f}% in NCR.\nOFW income and education were the strongest predictors."),
	]
	y = 222
	for head, big, body in blocks:
		label(1290, y, head, size=19, weight="bold", color=t["muted"])
		label(1290, y + 34, big, size=64, weight="bold", color=t["accent"])
		label(1290, y + 120, body, size=20)
		y += 232

	label(80, 1018, "Estimates use survey weights. Savings refers to families that saved any part of the remittance (C27), "
		f"not the OFW's own savings abroad. Each region has {min(region_n.values())}–{max(region_n.values())} households.",
		size=17, color=t["muted"])
	label(80, 1040, CREDIT, size=17, color=t["muted"])
	fig.savefig(out / f"{t['name']}.png", dpi=300, facecolor=t["background"])
	fig.savefig(out / f"{t['name']}.pdf", facecolor=t["background"])
	plt.close(fig)
