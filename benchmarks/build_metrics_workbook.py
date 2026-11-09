#!/usr/bin/env python3
"""Build benchmark_metrics.xlsx: full ESMFold2 pMHC-TCR benchmark workbook.

Sheets:
  1. 全部指标    -- all 56 complexes, every metric, ranked by mean pep-TCR
  2. TCR分组基线 -- per-TCR within-group baseline (cognate vs foreign)
  3. 指标区分度  -- positive/negative separation + AUC per metric, threshold stats
  4. 说明        -- metric definitions, provenance, caveats
Values-only (no formulas); styled via the xlsx skill design system.
"""
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

XLSX_SKILL_DIR = "/home/zhuhb/.zcode/cli/plugins/cache/zcode-plugins-official/spreadsheets/0.2.1/skills/xlsx"
for sub in [XLSX_SKILL_DIR, os.path.join(XLSX_SKILL_DIR, "templates")]:
    if sub not in sys.path:
        sys.path.insert(0, sub)
from base import *  # noqa: F401,F403 - design tokens & style factories

from openpyxl import Workbook
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles import Font

REPO = Path("/disk1/zhuhb/useful_tools/esmfold2")
LEGACY_POS = {"1G4_NYESO1_HLA-A2"}
SOURCES = [
    ("out_test/results.csv", "example_pmhc_tcr.csv", "批次1"),
    ("out_negatives/results.csv", "negatives_pmhc_tcr.csv", "批次2"),
    ("out_expanded/results.csv", "expanded_pmhc_tcr.csv", "批次3"),
    ("out_cohort3/results.csv", "cohort3.csv", "批次4"),
]
SCALE = dict(start_type="min", start_color="F8696B",
             mid_type="percentile", mid_value=50, mid_color="FFEB84",
             end_type="max", end_color="63BE7B")


def is_pos(pid: str) -> bool:
    return pid.startswith("POS_") or pid in LEGACY_POS


def load() -> list[dict]:
    rows = []
    for res_csv, in_csv, cohort in SOURCES:
        inputs = {r["id"]: r for r in csv.DictReader(open(REPO / in_csv))}
        for r in csv.DictReader(open(REPO / res_csv)):
            src = inputs.get(r["id"])
            a, b = float(r["iptm_peptide_tcra"]), float(r["iptm_peptide_tcrb"])
            rows.append({
                "id": r["id"], "kind": "阳性" if is_pos(r["id"]) else "阴性",
                "peptide": src["peptide"] if src else "",
                "cohort": cohort,
                "iptm": float(r["iptm"]), "ptm": float(r["ptm"]),
                "plddt": float(r["mean_plddt"]),
                "pep_mhc": float(r["iptm_peptide_mhc"]),
                "pep_tcra": a, "pep_tcrb": b,
                "mhc_tcra": float(r["iptm_mhc_tcra"]),
                "mhc_tcrb": float(r["iptm_mhc_tcrb"]),
                "tcra_tcrb": float(r["iptm_tcra_tcrb"]),
                "mean_ptcr": (a + b) / 2,
                "composite": ((a + b) / 2 + float(r["iptm"])) / 2,
                "nres": int(r["num_residues"]), "secs": float(r["fold_seconds"]),
            })
    return rows


def write_table(ws, title, headers, rows, num_cols, fmt="0.000", note=None):
    """Write one styled table at B4. num_cols: set of 0-based col indices that are numeric."""
    last_col = len(headers) + 1
    setup_sheet(ws, title=title, last_col=last_col)
    for j, h in enumerate(headers, start=2):
        ws.cell(row=4, column=j, value=h)
    style_header_row(ws, 4, 2, last_col)
    for i, row in enumerate(rows):
        rn = 5 + i
        for j, v in enumerate(row, start=2):
            c = ws.cell(row=rn, column=j, value=v)
            if (j - 2) in num_cols and isinstance(v, (int, float)):
                c.alignment = align_number()
                c.number_format = fmt if "耗时" not in str(headers[j - 2]) else "0.0"
        style_data_row(ws, rn, 2, last_col, i)
    if note:
        c = ws.cell(row=5 + len(rows) + 1, column=2, value=note)
        c.font = font_caption()
    auto_fit_columns(ws, min_width=8, max_width=30, header_row=4, data_start_row=5)
    auto_fit_row_heights(ws, header_row=4, data_start_row=5)
    return 5 + len(rows) - 1  # last data row


def main() -> None:
    rows = load()
    rows.sort(key=lambda r: -r["mean_ptcr"])
    pos = [r for r in rows if r["kind"] == "阳性"]
    neg = [r for r in rows if r["kind"] == "阴性"]

    wb = Workbook()

    # ---- Sheet 1: 全部指标 -------------------------------------------------
    ws = wb.active
    ws.title = "全部指标"
    headers = ["id", "类别", "抗原肽", "全局ipTM", "全局pTM", "平均pLDDT",
               "ipTM pep-MHC", "ipTM pep-TCRα", "ipTM pep-TCRβ",
               "ipTM MHC-TCRα", "ipTM MHC-TCRβ", "ipTM TCRα-TCRβ",
               "mean pep-TCR", "总残基数", "耗时(s)"]
    table = [[r["id"], r["kind"], r["peptide"], r["iptm"], r["ptm"], r["plddt"],
              r["pep_mhc"], r["pep_tcra"], r["pep_tcrb"], r["mhc_tcra"],
              r["mhc_tcrb"], r["tcra_tcrb"], r["mean_ptcr"], r["nres"], r["secs"]]
             for r in rows]
    last = write_table(ws, f"ESMFold2 pMHC–TCR 基准：全部指标（{len(rows)} 个复合体，按 mean pep-TCR 排序）",
                       headers, table, num_cols=set(range(3, 15)), note="来源: out_test / out_negatives / out_expanded / out_cohort3 的 results.csv")
    # accent colors for 类别 column (col C=3)
    for rn in range(5, last + 1):
        c = ws.cell(row=rn, column=3)
        c.font = Font(name=FONT_NAME, size=11,
                      color=ACCENT_POSITIVE if c.value == "阳性" else ACCENT_NEGATIVE)
    # color scales on the two headline metrics
    for col in ("E", "N"):
        ws.conditional_formatting.add(f"{col}5:{col}{last}", ColorScaleRule(**SCALE))

    # ---- Sheet 2: TCR分组基线 ----------------------------------------------
    groups = defaultdict(list)
    for r in rows:
        groups[r["id"]].append(r)  # id share only within same input csv; regroup below
    # regroup properly by (tcralpha, tcrbeta) via source csvs
    tcr_of = {}
    for _, in_csv, _ in SOURCES:
        for r in csv.DictReader(open(REPO / in_csv)):
            tcr_of[r["id"]] = (r["tcralpha"], r["tcrbeta"])
    groups = defaultdict(list)
    for r in rows:
        groups[tcr_of[r["id"]]].append(r)

    def group_label(rs):
        cog = next((r for r in rs if r["kind"] == "阳性"), rs[0])
        return cog["id"].replace("POS_", "").replace("_HLA-A2", "").replace("1G4_NYESO1", "1G4")

    grows = []
    ok = bad = 0
    for _, rs in groups.items():
        if len(rs) < 2:
            continue
        cog = [r["mean_ptcr"] for r in rs if r["kind"] == "阳性"]
        ng = [r["mean_ptcr"] for r in rs if r["kind"] == "阴性"]
        if not cog or not ng:
            continue
        c = cog[0]
        rank = 1 + sum(1 for n in ng if n > c)
        good = rank == 1
        ok, bad = ok + good, bad + (not good)
        grows.append([group_label(rs), len(rs), c, min(ng), max(ng), rank, "正确" if good else "错误"])
    grows.sort(key=lambda g: -g[2])
    ws2 = wb.create_sheet("TCR分组基线")
    last2 = write_table(ws2, f"按 TCR 分组的组内基线（mean pep-TCR；认知排第 1 的组：{ok}/{ok + bad}）",
                        ["TCR", "复合体数", "认知分", "阴性最低", "阴性最高", "认知组内排名", "组内排序"],
                        grows, num_cols={1, 2, 3, 4, 5})
    for rn in range(5, last2 + 1):
        c = ws2.cell(row=rn, column=8)
        c.font = Font(name=FONT_NAME, size=11,
                      color=ACCENT_POSITIVE if c.value == "正确" else ACCENT_NEGATIVE)

    # ---- Sheet 3: 指标区分度 ------------------------------------------------
    def auc(p, n):
        w = t = 0
        for x in p:
            for y in n:
                w += x > y
                t += x == y
        return (w + 0.5 * t) / (len(p) * len(n))

    metrics = [("全局 ipTM", "iptm"), ("全局 pTM", "ptm"), ("平均 pLDDT", "plddt"),
               ("ipTM pep-MHC", "pep_mhc"), ("ipTM pep-TCRα", "pep_tcra"),
               ("ipTM pep-TCRβ", "pep_tcrb"), ("ipTM MHC-TCRα", "mhc_tcra"),
               ("ipTM MHC-TCRβ", "mhc_tcrb"), ("ipTM TCRα-TCRβ", "tcra_tcrb"),
               ("mean pep-TCR", "mean_ptcr"), ("复合分(0.5+0.5)", "composite")]
    mrows = []
    for label, key in metrics:
        p = [r[key] for r in pos]
        n = [r[key] for r in neg]
        mrows.append([label, min(p), max(n), min(p) - max(n), auc(p, n)])
    mrows.sort(key=lambda m: -m[4])
    thr = 0.60
    tp = sum(1 for r in pos if r["mean_ptcr"] > thr)
    fp = sum(1 for r in neg if r["mean_ptcr"] > thr)
    ws3 = wb.create_sheet("指标区分度")
    last3 = write_table(ws3, "各指标阳/阴区分度（22 阳性 / 34 阴性，AUC 降序）",
                        ["指标", "阳性最低", "阴性最高", "间隙(阳-阴)", "AUC"],
                        mrows, num_cols={1, 2, 3, 4})
    r0 = last3 + 3
    c = ws3.cell(row=r0, column=2, value=f"阈值规则：mean pep-TCR > {thr}（高置信结合过滤器）")
    c.font = font_subheader()
    trows = [
        ["真阳性(敏感性)", f"{tp}/{len(pos)} = {tp / len(pos):.0%}"],
        ["假阳性", f"{fp}/{len(neg)} = {fp / len(neg):.0%}"],
        ["精确率", f"{tp}/{tp + fp} = {tp / (tp + fp):.0%}"],
        ["注意", "只可作阳性过滤器；不能反推『不结合』（约一半真阳性低于阈值）"],
    ]
    for j, h in enumerate(["规则结果", "数值"], start=2):
        ws3.cell(row=r0 + 1, column=j, value=h)
    style_header_row(ws3, r0 + 1, 2, 3)
    for i, row in enumerate(trows):
        for j, v in enumerate(row, start=2):
            ws3.cell(row=r0 + 2 + i, column=j, value=v)
        style_data_row(ws3, r0 + 2 + i, 2, 3, i)
    auto_fit_columns(ws3, min_width=8, max_width=44, header_row=4, data_start_row=5)
    auto_fit_row_heights(ws3, header_row=4, data_start_row=5)

    # ---- Sheet 4: 说明 ------------------------------------------------------
    ws4 = wb.create_sheet("说明")
    notes = [
        ("基准构成", "56 个 pMHC–TCR 复合体（22 阳性 = 晶体结构验证的认知配对；34 阴性 = 交叉配对/换肽/跨 MHC 的非认知配对），来自 4 个批次。"),
        ("链顺序", "A=peptide, B=MHC, C=TCRα, D=TCRβ；pair_chains_iptm 矩阵按此顺序索引。"),
        ("全局 ipTM", "整个复合体的界面 pTM（所有跨链界面），受 pMHC 自组装影响，跨 TCR 基线差异大。"),
        ("ipTM pep-TCRα/β", "peptide 残基与 TCRα/TCRβ 残基间的链对 ipTM，直接反映 TCR–肽界面识别。"),
        ("mean pep-TCR", "(ipTM pep-TCRα + ipTM pep-TCRβ)/2，本基准中最好的单一排序指标。"),
        ("pLDDT", "模型逐残基置信度（0-100），所有复合体均在 86-93，说明差异不来自整体折叠质量。"),
        ("核心结论", "排序用 mean pep-TCR，不要用全局 ipTM；绝对分数跨 TCR 不可比（每个 TCR 有自己的基线）；跨 TCR 比较前需先用阴性肽校准基线。"),
        ("已知盲区", "低基线 TCR（3GSN/3DXA/7RM4/LC13，认知分 0.33-0.41）模型对其真实界面识别差；高基线 TCR（4JFF/6DKP/5NME，错肽仍 0.57-0.63）对肽置换不敏感。"),
        ("注意事项", "阴性为计算交叉配对而非实验验证；单样本扩散采样，边界差值<0.05 的判断需多样本复核（--num-diffusion-samples 5）；MHC 均未加 β2m。"),
        ("复现", "uv run python run_pmhc_tcr_iptm.py -i <csv> -o <dir> -g <gpu>；汇总: uv run python summarize_benchmark.py"),
    ]
    setup_sheet(ws4, title="说明与注意事项", last_col=3)
    for j, h in enumerate(["项目", "内容"], start=2):
        ws4.cell(row=4, column=j, value=h)
    style_header_row(ws4, 4, 2, 3)
    for i, (k, v) in enumerate(notes):
        ws4.cell(row=5 + i, column=2, value=k)
        ws4.cell(row=5 + i, column=3, value=v)
        style_data_row(ws4, 5 + i, 2, 3, i)
    ws4.column_dimensions["C"].width = 100
    auto_fit_row_heights(ws4, header_row=4, data_start_row=5)

    wb.properties.creator = "Z.ai"
    out = REPO / "benchmark_metrics.xlsx"
    wb.save(out)
    print(f"wrote {out} | rows={len(rows)} pos={len(pos)} neg={len(neg)} | within-TCR correct {ok}/{ok + bad}")


if __name__ == "__main__":
    main()
