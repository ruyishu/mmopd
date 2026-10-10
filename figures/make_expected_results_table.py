"""Draw the hypothetical-target tables. Red cells are hopes, not measurements."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

fp = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
fm.fontManager.addfont(fp)
plt.rcParams["font.family"] = fm.FontProperties(fname=fp).get_name()
plt.rcParams["axes.unicode_minus"] = False

RED = "#dc2626"
BLACK = "#111827"
HEADER_BG = "#f3f4f6"
PUB_BG = "#ffffff"
TGT_BG = "#fff1f2"

# None means a published or empty cell (black). A string in TARGETS is red.
MAIN_ROWS = [
    ["教师 Qwen3-VL-4B-Instruct", "83.9", "—", "—", "881", "—", "—", "—"],
    ["Qwen3-1.7B-Base（纯文本）", "—", "—", "—", "—", "62.63", "75.44", "—"],
    ["B1 LLaVA 式 SigLIP+MLP+SFT", "72.4", "65.4", "88.3", "628", "59.10", "71.20", "86.3%"],
    ["B2 无编码器 + SFT", "57.4", "53.6", "83.4", "405", "57.40", "68.80", "68.4%"],
    ["B3 无编码器 + 离线 KD", "64.2", "60.2", "85.2", "488", "59.80", "71.50", "76.5%"],
    ["本文 无编码器 + OPD", "71.6", "66.8", "87.2", "536", "62.10", "74.90", "85.3%"],
]
MAIN_PUBLISHED = {0, 1}

SCALE_ROWS = [
    ["B2 无编码器 + SFT", "30.5", "39.2", "46.0"],
    ["本文 无编码器 + OPD", "45.5", "59.4", "67.5"],
]

ABLATION_ROWS = [
    ["OPD，教师看图，短预热", "71.6", "62.10"],
    ["SFT，短预热", "57.4", "57.40"],
    ["离线 KD，短预热", "64.2", "59.80"],
    ["OPD，教师只看文字描述", "53.8", "61.40"],
    ["OPD，预热 0 步", "27.1", "46.20"],
]


def paint(table, published_rows, font_size):
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d1d5db")
        cell.set_linewidth(0.6)
        text = cell.get_text()
        text.set_fontsize(font_size)
        if row == 0:
            cell.set_facecolor(HEADER_BG)
            text.set_color(BLACK)
            text.set_weight("bold")
            continue
        data_row = row - 1
        if data_row in published_rows:
            cell.set_facecolor(PUB_BG)
            text.set_color(BLACK)
        else:
            cell.set_facecolor(TGT_BG)
            text.set_color(RED if col > 0 else BLACK)
            if col > 0:
                text.set_weight("medium")
        if col == 0:
            text.set_ha("left")
            cell.PAD = 0.04


def draw(ax, col_labels, rows, published_rows, font_size, title, note):
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=13, color=BLACK, pad=10, fontweight="bold")
    table = ax.table(cellText=rows, colLabels=col_labels, loc="upper center", cellLoc="center")
    table.auto_set_font_size(False)
    table.scale(1, 1.45)
    paint(table, published_rows, font_size)
    # Left-align the first column a bit more by widening via auto.
    ax.text(
        0.0,
        -0.08,
        note,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        color=BLACK,
    )


def main():
    fig = plt.figure(figsize=(16.2, 11.6), dpi=140, facecolor="white")
    fig.suptitle(
        "演示设定：教师 Qwen3-VL-4B-Instruct，学生 Qwen3-1.7B",
        fontsize=16,
        fontweight="bold",
        color=BLACK,
        y=0.975,
    )
    fig.text(
        0.5,
        0.935,
        "红色=预期目标，黑色=已公布真实结果",
        ha="center",
        va="center",
        fontsize=13,
        color=BLACK,
    )
    fig.text(
        0.5,
        0.905,
        "红色数字是假设目标，不是实验结果。相对教师比例只按 MMBench / 83.9。",
        ha="center",
        va="center",
        fontsize=11,
        color="#4b5563",
    )

    gs = fig.add_gridspec(3, 1, height_ratios=[1.55, 0.72, 1.15], left=0.04, right=0.98, top=0.87, bottom=0.04, hspace=0.38)
    main_ax = fig.add_subplot(gs[0])
    scale_ax = fig.add_subplot(gs[1])
    abl_ax = fig.add_subplot(gs[2])

    draw(
        main_ax,
        ["模型", "MMBench", "TextVQA", "POPE", "OCRBench", "MMLU", "GSM8K", "相对教师比例"],
        MAIN_ROWS,
        MAIN_PUBLISHED,
        9,
        "主表（同数据预算下的全量目标）",
        "要证明的形态：无编码器 OPD 在综合题上靠近同数据的有编码器 SFT，OCR 更弱，文本几乎保住底座。",
    )
    draw(
        scale_ax,
        ["模型", "MMBench 1%", "MMBench 5%", "MMBench 10%"],
        SCALE_ROWS,
        set(),
        10,
        "数据缩放（只有 MMBench，全部是预期）",
        "要证明的形态：数据很少时，OPD 的 MMBench 比 SFT 抬得更快。",
    )
    draw(
        abl_ax,
        ["设置", "MMBench", "MMLU"],
        ABLATION_ROWS,
        set(),
        10,
        "消融（全部是预期）",
        "要证明的形态：教师不看图，或预热为零，OPD 就不成立。",
    )

    fig.savefig(
        "/workspace/figures/expected_results_table.png",
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.25,
    )


if __name__ == "__main__":
    main()
