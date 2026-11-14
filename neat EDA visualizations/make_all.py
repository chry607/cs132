from bars import make_h1, make_h2, make_h3
from data import ROOT, load_data
from models import make_rq2, make_rq3
from nutshell import make_nutshell

OUT = ROOT / "outputs" / "slides"


def main():
	OUT.mkdir(parents=True, exist_ok=True)
	data = load_data()
	make_h1(data, OUT)
	make_h2(data, OUT)
	make_h3(data, OUT)
	make_rq2(data, OUT)
	make_rq3(data, OUT)
	make_nutshell(OUT, "deck")
	make_nutshell(OUT, "course")
	print(f"Slides saved to {OUT}")


if __name__ == "__main__":
	main()
