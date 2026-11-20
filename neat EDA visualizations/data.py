from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "PHL-PSA-SOF-2024-PUF 2" / "SOF PUF 2024.CSV"

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

MARITAL_GROUPS = {
	1: "Single", 2: "Married", 3: "Married",
	4: "Previously married", 5: "Previously married",
	6: "Previously married", 7: "Previously married",
}
MARITAL_ORDER = ["Single", "Married", "Previously married"]

DESTINATION_RANGES = {
	"Africa": [100],
	"East Asia": [211, 212, 213, 214],
	"Southeast / South Central Asia": [221, 222, 223],
	"Western Asia (Middle East)": [231, 232, 233, 234, 235],
	"Australia": [300],
	"Europe": [400],
	"Americas": [500],
	"Other": [600],
}
DESTINATION_LABELS = {
	code: label for label, codes in DESTINATION_RANGES.items() for code in codes
}
DESTINATION_ORDER = list(DESTINATION_RANGES.keys())

REMIT_MODE_GROUPS = {
	1: "Formal Bank", 2: "OTC", 3: "OTC", 4: "Door-to-door", 5: "OTC", 6: "OTC",
}
REMIT_MODE_ORDER = ["Formal Bank", "Door-to-door", "OTC"]

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
		"RQ3_SEX", "RQ4_AGE", "RQ7_MSTAT", "RQ8_HGRADE", "RQ9_2DGT", "RQ10_REASON",
		"RQ12_CTRY", "RQ17_AVEMINC", "RQ26_MODE", "RQ273P_SAVINGS", "RP6M_IND",
		"RREG", "RURB2020", "RSWGT",
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
	data["marital_status"] = data["RQ7_MSTAT"].map(MARITAL_GROUPS)
	data["destination_region"] = data["RQ12_CTRY"].map(
		lambda value: DESTINATION_LABELS.get(int(value)) if pd.notna(value) else np.nan
	)
	data["savings_pct"] = data["RQ273P_SAVINGS"].where(data["RQ273P_SAVINGS"].between(0, 100))
	data["any_savings"] = data["savings_pct"].gt(0).where(data["savings_pct"].notna())
	data["any_savings_num"] = data["any_savings"].astype(float)
	data["mode"] = data["RQ26_MODE"].map(
		lambda value: REMIT_MODE_GROUPS.get(int(value)) if pd.notna(value) else np.nan
	)
	data["mode_code"] = data["mode"].map({"Formal Bank": 0, "Door-to-door": 1, "OTC": 2})
	data["non_bank_num"] = data["mode"].ne("Formal Bank").astype(float).where(data["mode"].notna())
	return data

def wmean(values: pd.Series, weights: pd.Series) -> float:
	valid = values.notna() & weights.notna() & weights.gt(0)
	return float(np.average(values[valid], weights=weights[valid])) if valid.any() else np.nan
