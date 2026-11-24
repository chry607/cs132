import base64
import io
import json
import math
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors, font_manager
from matplotlib.collections import PolyCollection
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
from shapely.geometry import shape
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

from data import DATA_PATH, ROOT

THEMES = {
	"deck": dict(
		background="white", text="#0f1e3d", muted="#5b6880", divider="#e3e8f0",
		map_colors=["#dbe5f5", "#93b4e8", "#5b8ad8", "#02164d"], map_side="#c3cee0", outline="white", accent="#1e3a8a",
		font=["Roboto"], title=dict(y=50, size=50, va="top"),
		subtitle=dict(y=117, size=27, va="top"), name="nutshell"),
	"course": dict(
		background="#1B181C", text="#FFFFFF", muted="#C8C1CB", divider="#3A333D",
		map_colors=["#4B2362", "#9E3E7F", "#DE6B61", "#EDB081"], map_side="#49304F", outline="#D9C8DD", accent="#FFB000",
		font=["Roboto"], title=dict(y=50, size=50, va="top"), subtitle=dict(y=117, size=27, va="top"), name="nutshell_course"),
}

TITLE = "Across Philippine Regions, How OFWs Work and Remit"
SUBTITLE = "Contract work, family savings and remittance channels · 2024"
MAP_LEGEND = "Bank remittance use"
CREDIT = "Chart: CS 132 Group 4 · Boundaries: J. Faeldon (2023 vintage, MIT)"
LABELS = {
	14: ("Cordillera", 242, 255),
	13: ("NCR", 261, 445),
	17: ("MIMAROPA", 216, 748),
	7: ("Central Visayas", 994, 651),
	16: ("Caraga", 1084, 774),
	12: ("SOCCSKSARGEN", 957, 912),
}
DEPTH, TILT = 15, 0.86
MAP_TOP, MAP_H, MAP_CX = 198, 737, 726
BRICK_PERCENT, BRICK_WIDTH, PIXELS_PER_PERCENT = 5, 12, 3.0
GOLD, GOLD_SIDE, GOLD_TOP, GOLD_SEAM = "#FFB000", "#AE6911", "#FFE4A1", "#593409"


def load_assets():
	assets = ROOT / ".plain_map_assets"
	geometry = assets / "assets/geography/regions_2023.geojson"
	fonts = assets / "assets/fonts"
	if geometry.is_file() and all((fonts / f"Roboto-{name}.ttf").is_file()
	                              for name in ("Regular", "Bold")):
		return json.loads(geometry.read_text(encoding="utf-8"))
	assets.mkdir(exist_ok=True)
	nb = json.loads((ROOT / "nutshell_plain.ipynb").read_text(encoding="utf-8"))
	with zipfile.ZipFile(io.BytesIO(base64.b64decode(nb["metadata"]["map_assets"]))) as archive:
		for member in archive.infolist():
			if not (assets / member.filename).resolve().is_relative_to(assets.resolve()):
				raise ValueError("Invalid asset path")
		archive.extractall(assets)
	return json.loads(geometry.read_text(encoding="utf-8"))


def summarise():
	df = pd.read_csv(DATA_PATH, dtype=str, encoding="utf-8-sig")
	for c in ["RREG", "RSWGT", "RQ10_REASON", "RQ273P_SAVINGS", "RQ26_MODE"]:
		df[c] = pd.to_numeric(df[c].str.strip(), errors="coerce")
	df = df.loc[df.RSWGT.gt(0)].copy()
	df["contract"] = df.RQ10_REASON.eq(1).where(df.RQ10_REASON.between(1, 8))
	df["savings"] = df.RQ273P_SAVINGS.where(df.RQ273P_SAVINGS.between(0, 100))
	df["bank"] = df.RQ26_MODE.eq(1).where(df.RQ26_MODE.between(1, 6))

	def share(d, col, scale=100):
		d = d[d[col].notna()]
		return scale * np.average(d[col].astype(float), weights=d.RSWGT)

	return {
		int(r): {"bank": share(g, "bank"), "savings": share(g, "savings", scale=1),
		         "contract": share(g, "contract")}
		for r, g in df.groupby("RREG")
	}


def make_nutshell(out, theme="deck"):
	t = THEMES[theme]
	geometry = load_assets()
	regional = summarise()

	geoms = {r: unary_union([shape(f["geometry"]) for f in geometry["features"] if f["properties"]["region"] == r]) for r in regional}
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

	cmap = colors.LinearSegmentedColormap.from_list("bank_use", t["map_colors"])
	norm = colors.Normalize(vmin=0, vmax=100, clip=True)
	plt.rcParams.update({"font.family": t["font"], "pdf.fonttype": 42, "svg.fonttype": "none"})
	fig = plt.figure(figsize=(6.4, 3.6), dpi=300, facecolor=t["background"])
	ax = fig.add_axes([0, 0, 1, 1])
	ax.set_xlim(0, 1920)
	ax.set_ylim(1080, 0)
	ax.axis("off")

	font_dir = ROOT / ".plain_map_assets/assets/fonts"
	fonts = {
		"normal": font_manager.FontProperties(family="Roboto", fname=font_dir / "Roboto-Regular.ttf"),
		"bold": font_manager.FontProperties(family="Roboto", weight="bold", fname=font_dir / "Roboto-Bold.ttf"),
	}

	def label(x, y, text, size=24, weight="normal", ha="left", color=t["text"], va="top"):
		return ax.text(x, y, text, fontsize=size * 72 / 300, color=color,
		               fontproperties=fonts[weight], ha=ha, va=va, linespacing=1.26, zorder=6)

	label(80, t["title"]["y"], TITLE, size=t["title"]["size"], weight="bold", va=t["title"]["va"])
	label(80, t["subtitle"]["y"], SUBTITLE, size=t["subtitle"]["size"], color=t["muted"], va=t["subtitle"]["va"])

	ax.plot([80, 1840], [172, 172], color=t["divider"], lw=0.55)
	ax.plot([1375, 1375], [219, 938], color=t["divider"], lw=0.55)

	for depth in range(DEPTH, 0, -3):
		for g in geoms.values():
			ax.add_patch(PathPatch(path(g, depth), facecolor=t["map_side"], edgecolor=t["map_side"], linewidth=0.15, zorder=1))
	for r, g in geoms.items():
		ax.add_patch(PathPatch(path(g), facecolor=cmap(norm(regional[r]["bank"])), edgecolor=t["outline"], linewidth=0.25, zorder=2))

	centers = {r: project([g.centroid.x, g.centroid.y]) for r, g in geoms.items()}

	def gold_bricks(x, y, percent):
		"""Draw square-front bricks with a slight extrusion and visible seams."""
		unit = BRICK_PERCENT
		scale = PIXELS_PER_PERCENT
		width = BRICK_WIDTH
		depth, rise = width * 0.32, width * 0.22
		left, right = x - (width + depth) / 2, x + (width - depth) / 2
		layers = np.append(np.arange(0, percent, unit), percent) * scale
		if len(layers) < 2:
			return

		front, sides = [], []
		for bottom, top in zip(layers[:-1], layers[1:]):
			base, ledge = y - bottom, y - top
			front.append([(left, base), (left, ledge), (right, ledge), (right, base)])
			sides.append([(right, base), (right, ledge),
						  (right + depth, ledge - rise), (right + depth, base - rise)])

		top = y - layers[-1]
		sides.append([(left, top), (right, top),
					  (right + depth, top - rise), (left + depth, top - rise)])
		faces = (
			(front, GOLD),
			(sides, [GOLD_SIDE] * (len(sides) - 1) + [GOLD_TOP]),
		)
		for polygons, fills in faces:
			ax.add_collection(PolyCollection(
				polygons, facecolors=fills, edgecolors=GOLD_SEAM,
				linewidths=0.29, joinstyle="miter", zorder=4,
			))

	for region in sorted(centers, key=lambda r: centers[r][1]):
		gold_bricks(*centers[region], regional[region]["savings"])

	region_labels = []
	for r, (name, tx, ty) in LABELS.items():
		artist = label(tx, ty, f"{name} ({regional[r]['contract']:.1f}%)", size=23)
		region_labels.append((r, artist))

	left, right = 1416, 1840
	label(left, 232, MAP_LEGEND, size=25, weight="bold")
	label(left, 272, "Map color · % using a bank among\nthose who reported a remittance channel",
	      size=22, color=t["muted"])
	ax.imshow(np.linspace(0, 100, 400).reshape(1, -1), extent=(left, right, 380, 357),
	          cmap=cmap, norm=norm, aspect="auto", zorder=1)
	for x, value, ha in [(left, "0%", "left"), ((left + right) / 2, "50%", "center"), (right, "100%", "right")]:
		label(x, 393, value, size=22, color=t["muted"], ha=ha)

	label(left, 489, "Receiving-family savings", size=25, weight="bold")
	label(left, 530, "Gold bricks · Average % of received\nremittances set aside by families",
	      size=22, color=t["muted"])
	gold_bricks(left + 9, 634, BRICK_PERCENT)
	label(left + 43, 605, f"1 full brick = {BRICK_PERCENT}% saved", size=22, weight="bold")
	label(left, 660, "A partial brick shows the remainder.", size=20, color=t["muted"])

	label(left, 725, "Contract-worker share", size=25, weight="bold")
	label(left, 767, "Region labels · % who left as\ncontract workers among surveyed\n"
	      "overseas Filipinos in each labeled region", size=22, color=t["muted"])

	label(80, 982, "Survey-weighted regional estimates. Each measure uses the records "
	      "that report it. Savings are allocated by receiving families.", size=19, color=t["muted"])
	label(80, 1014, "Data: PSA Survey on Overseas Filipinos 2024 Public Use File", size=18, color=t["muted"])
	label(1840, 1014, CREDIT, size=18, ha="right", color=t["muted"])

	fig.canvas.draw()
	renderer = fig.canvas.get_renderer()
	for r, artist in region_labels:
		x, y = centers[r]
		box = artist.get_window_extent(renderer).transformed(ax.transData.inverted())
		x_edge = np.clip(x, min(box.x0, box.x1), max(box.x0, box.x1))
		y_edge = np.clip(y, min(box.y0, box.y1) - 3, max(box.y0, box.y1) + 3)
		ax.plot([x, x_edge], [y, y_edge], color=t["muted"], linewidth=0.32, zorder=5)

	fig.savefig(out / f"{t['name']}.png", dpi=300, facecolor=t["background"])
	fig.savefig(out / f"{t['name']}.pdf", facecolor=t["background"])
	plt.close(fig)
