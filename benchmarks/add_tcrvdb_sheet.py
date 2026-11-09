#!/usr/bin/env python3
"""Append TCRvdb validation sheets to benchmark_metrics.xlsx."""
import csv
import math
import os
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

XLSX_SKILL_DIR = "/home/zhuhb/.zcode/cli/plugins/cache/zcode-plugins-official/spreadsheets/0.2.1/skills/xlsx"
for sub in [XLSX_SKILL_DIR, os.path.join(XLSX_SKILL_DIR, "templates")]:
    if sub not in sys.path:
        sys.path.insert(0, sub)
from base import *  # noqa: F401,F403

from openpyxl import load_workbook
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles import Font

REPO = Path("/disk1/zhuhb/useful_tools/esmfold2")
SCALE = dict(start_type="min", start_color="F8696B",
             mid_type="percentile", mid_value=50, mid_color="FFEB84",
             end_type="max", end_color="63BE7B")

meta = {r["id"]: r for r in csv.DictReader(open(REPO / "tcrvdb_validation_meta.csv"))}
rows = []
for r in csv.DictReader(open(REPO / "out_tcrvdb" / "results.csv")):
    m = meta[r["id"]]
    a, b = float(r["iptm_peptide_tcra"]), float(r["iptm_peptide_tcrb"])
    rows.append({"id": r["id"], "lab": m["label"], "ep": m["epitope"], "hla": m["hla"],
                 "score": m["vdjdb_score"], "iptm": float(r["iptm"]), "plddt": float(r["mean_plddt"]),
                 "pep_mhc": float(r["iptm_peptide_mhc"]), "pep_tcra": a, "pep_tcrb": b, "mpt": (a + b) / 2})

by = defaultdict(lambda: defaultdict(list))
for r in rows:
    by[r["ep"]][r["lab"]].append(r)


def mw_p(pos, neg):
    u = sum(1 for x in pos for y in neg if x > y) + 0.5 * sum(1 for x in pos for y in neg if x == y)
    n1, n2 = len(pos), len(neg)
    z = (u - n1 * n2 / 2) / math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    return 0.5 * (1 + math.erf(-abs(z) / math.sqrt(2)))


wb = load_workbook(REPO / "benchmark_metrics.xlsx")

# ---- sheet: TCRvdb汇总 ----
if "TCRvdb汇总" in wb.sheetnames:
    del wb.sheetnames["TCRvdb汇总" and wb.sheetnames.index("TCRvdb汇总")]
ws = wb.create_sheet("TCRvdb汇总", 0)
last_col = 8
setup_sheet(ws, title="TCRvdb 实验标签验证（120 样本 = 5 表位 × 12 阳 + 12 阴，mean pep-TCR）", last_col=last_col)
headers = ["表位", "HLA", "阳性中位", "阴性中位", "AUC", "p 值", "阴性最大值", "该阈值下阳性召回"]
for j, h in enumerate(headers, start=2):
    ws.cell(row=4, column=j, value=h)
style_header_row(ws, 4, 2, last_col)
allp, alln = [], []
for i, (ep, g) in enumerate(sorted(by.items())):
    p = [x["mpt"] for x in g["pos"]]
    n = [x["mpt"] for x in g["neg"]]
    allp += p; alln += n
    u = sum(1 for x in p for y in n if x > y) + 0.5 * sum(1 for x in p for y in n if x == y)
    auc = u / (len(p) * len(n))
    pv = mw_p(p, n)
    nmax = max(n)
    rec = sum(1 for x in p if x > nmax) / len(p)
    vals = [ep, g["pos"][0]["hla"], st.median(p), st.median(n), auc, pv, nmax, rec]
    for j, v in enumerate(vals, start=2):
        c = ws.cell(row=5 + i, column=j, value=round(v, 4) if isinstance(v, float) else v)
        if j >= 4:
            c.alignment = align_number()
            c.number_format = "0.000" if j in (4, 5, 6, 8) else ("0.0000" if j == 7 else "0%")
    style_data_row(ws, 5 + i, 2, last_col, i)
# pooled row
u = sum(1 for x in allp for y in alln if x > y) + 0.5 * sum(1 for x in allp for y in alln if x == y)
vals = ["汇总", "A1+A2", st.median(allp), st.median(alln), u / (len(allp) * len(alln)), mw_p(allp, alln), None, None]
r0 = 5 + len(by)
for j, v in enumerate(vals, start=2):
    c = ws.cell(row=r0, column=j, value=round(v, 4) if isinstance(v, float) else v)
    if j >= 4 and isinstance(v, float):
        c.alignment = align_number(); c.number_format = "0.000"
style_total_row(ws, r0, 2, last_col)
note = ("标签来源: TCRvdb/VDJdb 实验筛选（pos=激活, neg=筛选阴性, ambiguous 已排除）；"
        "阳性样本偏好高 vdjdb_score（44/60 ≥1）。注意『阴性最大值阈值』召回 0-67%：每层有 1-3 个重叠样本，"
        "该指标适合排序，不适合单一阈值二分。")
c = ws.cell(row=r0 + 2, column=2, value=note)
c.font = font_caption()
auto_fit_columns(ws, min_width=9, max_width=30, header_row=4, data_start_row=5)
auto_fit_row_heights(ws, header_row=4, data_start_row=5)

# ---- sheet: TCRvdb验证明细 ----
if "TCRvdb验证明细" in wb.sheetnames:
    del wb["TCRvdb验证明细"]
ws2 = wb.create_sheet("TCRvdb验证明细", 1)
headers2 = ["id", "标签", "表位", "HLA", "vdjdb_score", "全局ipTM", "平均pLDDT",
            "ipTM pep-MHC", "ipTM pep-TCRα", "ipTM pep-TCRβ", "mean pep-TCR"]
last2 = len(headers2) + 1
setup_sheet(ws2, title=f"TCRvdb 验证明细（{len(rows)} 行，按表位/标签/分数排序）", last_col=last2)
for j, h in enumerate(headers2, start=2):
    ws2.cell(row=4, column=j, value=h)
style_header_row(ws2, 4, 2, last2)
srows = sorted(rows, key=lambda r: (r["ep"], r["lab"], -r["mpt"]))
for i, r in enumerate(srows):
    vals = [r["id"], r["lab"], r["ep"], r["hla"], float(r["score"]), r["iptm"], r["plddt"],
            r["pep_mhc"], r["pep_tcra"], r["pep_tcrb"], r["mpt"]]
    rn = 5 + i
    for j, v in enumerate(vals, start=2):
        c = ws2.cell(row=rn, column=j, value=v)
        if j >= 6:
            c.alignment = align_number()
            c.number_format = "0.000"
    style_data_row(ws2, rn, 2, last2, i)
    lab_c = ws2.cell(row=rn, column=3)
    lab_c.font = Font(name=FONT_NAME, size=11,
                      color=ACCENT_POSITIVE if r["lab"] == "pos" else ACCENT_NEGATIVE)
last_r = 4 + len(srows)
for col in ("G", "L"):
    ws2.conditional_formatting.add(f"{col}5:{col}{last_r}", ColorScaleRule(**SCALE))
auto_fit_columns(ws2, min_width=9, max_width=24, header_row=4, data_start_row=5)
auto_fit_row_heights(ws2, header_row=4, data_start_row=5)

wb.properties.creator = "Z.ai"
wb.save(REPO / "benchmark_metrics.xlsx")
print(f"added sheets: {wb.sheetnames}")
