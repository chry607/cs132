import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ABOVE, BELOW, SAME = "#1e3a8a", "#5b8ad8", "#c4c4c4"
TRACK, INK, MUTED = "#f2f5fa", "#0f1e3d", "#5b6880"
FONTS = ["Avenir Next", "Helvetica Neue", "DejaVu Sans"]


def new_slide(title, subtitle):
	plt.rcParams["font.family"] = FONTS
	fig = plt.figure(figsize=(13.33, 7.5), facecolor="white")
	fig.text(0.04, 0.905, title, fontsize=24, weight="bold", color=INK, va="baseline")
	fig.text(0.04, 0.866, subtitle, fontsize=13, color=MUTED, va="baseline")
	return fig


def legend(fig, items):
	x = 0.04
	for color, label in items:
		fig.patches.append(plt.Rectangle((x, 0.797), 0.012, 0.022, color=color, transform=fig.transFigure))
		text = fig.text(x + 0.018, 0.808, label, fontsize=11.5, color=INK, va="center")
		x += 0.018 + text.get_window_extent(fig.canvas.get_renderer()).width / fig.bbox.width + 0.03


def centre_x(fig, ax):
	box = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
	return ax.get_position().x0 + ((1 - box.x1) - box.x0) / 2


def move_x(ax, x0):
	pos = ax.get_position()
	ax.set_position([x0, pos.y0, pos.width, pos.height])


def finish(fig, source, path):
	fig.text(0.04, 0.03, source, fontsize=8.5, color=MUTED, linespacing=1.15)
	fig.savefig(path, dpi=300, facecolor="white")
	plt.close(fig)
