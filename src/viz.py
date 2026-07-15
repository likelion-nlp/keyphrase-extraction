"""시각화 스타일 모듈: 전 노트북이 공유하는 팔레트·rcParams·차트 헬퍼.

원칙:
- 모델(엔티티)마다 고정 색 슬롯을 배정한다 — 필터/정렬이 바뀌어도 색은 따라간다.
- 크기(magnitude)는 단일 색상 램프(파랑 계열)로만 표현한다 (히트맵 등).
- 이중 y축은 쓰지 않는다 — 스케일이 다른 두 지표는 패널을 위아래로 나눈다.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# 검증된 categorical 팔레트 (고정 순서 — 순환 배정 금지)
PALETTE = [
    "#2a78d6",  # 1 blue
    "#1baf7a",  # 2 aqua
    "#eda100",  # 3 yellow
    "#008300",  # 4 green
    "#4a3aa7",  # 5 violet
    "#e34948",  # 6 red
    "#e87ba4",  # 7 magenta
    "#eb6834",  # 8 orange
]

# 모델별 고정 색 (모든 노트북·그림에서 동일)
MODEL_COLORS = {
    "Random": "#898781",
    "TF-IDF": "#eda100",
    "KeyBERT": "#1baf7a",
    "KeyBERT+MMR": "#1baf7a",
    "BART": "#4a3aa7",
    "KeyBART": "#2a78d6",
    "Hybrid": "#e34948",
    "Hybrid+MMR": "#eb6834",
}

# 단일 색상 sequential 램프 (히트맵·magnitude 전용)
SEQ_BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUES)

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"


def apply_style() -> None:
    """프로젝트 공통 matplotlib 스타일을 적용한다. 모든 노트북 첫 셀에서 호출."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": "#c3c2b7",
            "axes.linewidth": 1.0,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "text.color": INK,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "font.family": "sans-serif",
            "figure.dpi": 110,
            "figure.autolayout": True,
            "axes.prop_cycle": mpl.cycler(color=PALETTE),
        }
    )


def model_color(name: str) -> str:
    """모델명 → 고정 색. 등록되지 않은 모델은 muted 회색."""
    for key, color in MODEL_COLORS.items():
        if name.lower().startswith(key.lower()):
            return color
    return MUTED


def bar_with_labels(ax, labels, values, colors=None, fmt="{:.3f}", label_offset=0.01):
    """세로 막대 + 값 직접 라벨. colors 미지정 시 모델 고정 색."""
    if colors is None:
        colors = [model_color(l) for l in labels]
    bars = ax.bar(labels, values, color=colors, width=0.62, zorder=3)
    vmax = max(values) if len(values) else 1.0
    for rect, v in zip(bars, values):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + label_offset * vmax,
            fmt.format(v),
            ha="center", va="bottom", fontsize=9, color=INK_2,
        )
    ax.set_ylim(0, vmax * 1.15 if vmax > 0 else 1)
    ax.grid(axis="x", visible=False)
    return bars


def save_figure(fig, path) -> None:
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    print(f"[figure] saved: {path}")
