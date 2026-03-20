"""
MOSFET 开通瞬态过程波形示意图
- 无具体数值，仅做定性示意
- 三条波形合并在同一坐标系（归一化）
- 大字体标注
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

# ── 全局字体与样式 ──────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family": ["Microsoft YaHei", "SimHei", "sans-serif"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
    "figure.dpi": 150,
    "svg.fonttype": "none",
})

# ── 归一化参数（无量纲，仅控制形状）─────────────────────────
# 时间节点（归一化，均匀排布）
t0, t1, t2, t3, t4 = 0.0, 0.25, 0.50, 0.75, 1.00
t_end = 1.05

# 电压电流归一化幅值
V_GH  = 1.0       # 驱动高电平
V_GL  = -0.05      # 驱动低电平
V_th  = 0.18
V_GP  = 0.35
V_DD  = 0.88       # 母线电压
V_DSon = 0.02
I_L   = 0.60       # 负载电流

g_m   = I_L / (V_GP - V_th)

# ── 构造波形数据 ───────────────────────────────────────────
N = 2000
t = np.linspace(t0, t_end, N)

vgs = np.zeros(N)
ids = np.zeros(N)
vds = np.zeros(N)

# 计算第一与第二阶段的统一指数参数，确保精确经过 (t0, V_GL), (t1, V_th), (t2, V_GP)
k_exp = (V_GP - V_th) / (V_th - V_GL)
B_exp = (V_th - V_GL) / (1 - k_exp)
A_exp = V_GL + B_exp
c_exp = -np.log(k_exp) / (t1 - t0)

for i, ti in enumerate(t):
    if ti <= t2:
        # 阶段一 & 阶段二：统一的 RC 充电指数曲线
        vgs_val = A_exp - B_exp * np.exp(-c_exp * (ti - t0))
        vgs[i] = vgs_val
        vds[i] = V_DD
        
        if ti <= t1:
            ids[i] = 0.0
        else:
            # 电流随 V_GS 指数上升而增长（满足物理公式）
            ids[i] = g_m * (vgs_val - V_th)
            
    elif ti <= t3:
        # 阶段三：Miller 平台
        frac = (ti - t2) / (t3 - t2)
        vgs[i] = V_GP
        ids[i] = I_L
        # 使用幂函数曲线 (1-frac)^3 完美模拟 C_GD 非线性造成的“先快后慢”下降，消除原来分段带来的折角
        vds[i] = V_DSon + (V_DD - V_DSon) * np.power(1.0 - frac, 3.0)
            
    else:
        # 阶段四：通态完成（重新开始指数充电直至 V_GH）
        # 物理学矫正：此时 V_DS 很低，C_GD 急剧增大导致总输入电容 C_iss 变大。
        # 因此，阶段四的充电斜率（dv/dt）必须明显小于在阶段二结束时（t2）的斜率。
        
        # 1. 计算 t2 结束时的充电斜率 slope_t2
        slope_t2 = B_exp * c_exp * np.exp(-c_exp * (t2 - t0))
        
        # 2. 设定 t3 刚开始时的斜率，反映电容增大的物理事实（比如减慢为原来的 1.5 倍）
        slope_t3 = slope_t2 / 1.5
        
        # 3. 反推阶段四指数函数的常数 c4
        # 因为在 t3 时，dv/dt = (V_GH - V_GP) * c4
        c4 = slope_t3 / (V_GH - V_GP)
        
        vgs[i] = V_GH - (V_GH - V_GP) * np.exp(-c4 * (ti - t3))
        ids[i] = I_L
        vds[i] = V_DSon

# ── 绘图 ──────────────────────────────────────────────────
fig, ax = plt.subplots(1, 1, figsize=(14, 7))

# 颜色
C_VGS  = "#2563EB"
C_ID   = "#DC2626"
C_VDS  = "#16A34A"
C_DASH = "#9CA3AF"

LW = 3.0
FONTSIZE_LABEL = 22
FONTSIZE_ANNO  = 18
FONTSIZE_STAGE = 16
FONTSIZE_AXIS  = 18

# 绘制三条波形
ax.plot(t, vgs, color=C_VGS, linewidth=LW, label=r"$V_{GS}$", zorder=5)
ax.plot(t, ids, color=C_ID,  linewidth=LW, label=r"$I_D$",  zorder=5)
ax.plot(t, vds, color=C_VDS, linewidth=LW, label=r"$V_{DS}$", zorder=5)

# 开关损耗交叠区域着色
mask_loss = (t >= t1) & (t <= t3)
ax.fill_between(t, -0.15, 1.1, where=mask_loss,
                alpha=0.08, color="#F59E0B", zorder=1,
                label="开关损耗区域")

# ── 关键电压/电流水平虚线 + 标注 ──────────────────────────
anno_x = -0.02  # 标注放在左侧

for val, label, color in [
    (V_GH,   r"$V_{GH}$",    C_VGS),
    (V_DD,   r"$V_{DD}$",     C_VDS),
    (I_L,    r"$I_L$",        C_ID),
    (V_GP,   r"$V_{GP}$",     C_VGS),
    (V_th,   r"$V_{th}$",     C_VGS),
    (V_DSon, r"$V_{DS(on)}$", C_VDS),
    (V_GL,   r"$V_{GL}$",     C_VGS),
]:
    ax.axhline(val, color=color, ls="--", lw=0.7, alpha=0.5, zorder=2)
    # 避免非常接近的标签重叠
    text_y = val
    if label == r"$V_{DS(on)}$":
        text_y = val + 0.02
    elif label == r"$V_{GL}$":
        text_y = val - 0.02
        
    ax.text(anno_x, text_y, label, ha="right", va="center", fontsize=FONTSIZE_ANNO,
            color=color, fontweight="bold")

# ── 时间节点竖直虚线 ──────────────────────────────────────
t_marks = [t0, t1, t2, t3, t4]
t_labels = [r"$t_0$", r"$t_1$", r"$t_2$", r"$t_3$", r"$t_4$"]

for tn in t_marks:
    ax.axvline(tn, color=C_DASH, ls=":", lw=1.0, zorder=2)

# ── 阶段名称标注（顶部）──────────────────────────────────
stage_names = [
    "阶段一\n开通延迟",
    "阶段二\n电流上升",
    "阶段三\nMiller平台",
    "阶段四\n通态完成",
]

for idx in range(len(stage_names)):
    x_center = (t_marks[idx] + t_marks[idx + 1]) / 2
    ax.text(x_center, 1.12, stage_names[idx],
            ha="center", va="bottom", fontsize=FONTSIZE_STAGE,
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec="#D1D5DB", lw=0.8))

# ── 坐标轴 ────────────────────────────────────────────────
ax.set_xlim(-0.12, t_end * 1.1)
ax.set_ylim(-0.25, 1.3)
# 去掉横轴“时间”标签

# 将y轴的主轴线放在x=0处，更有数学坐标系的感觉
ax.spines["left"].set_position(("data", 0))
# 将x轴的主轴线放在刚好包裹住V_GL的下方
ax.spines["bottom"].set_position(("data", -0.15))

# 使用标准的 x 轴刻度来显示时间标签，这样会自动放在横轴下方
# 去掉 t0，改为手动绘制，避免和主 y 轴线重叠
ax.set_xticks(t_marks[1:])
ax.set_xticklabels(t_labels[1:], fontsize=FONTSIZE_ANNO, fontweight="bold")
ax.text(t0 + 0.01, -0.165, r"$t_0$", ha="left", va="top", fontsize=FONTSIZE_ANNO, fontweight="bold")

# 隐藏y轴数值刻度（因为左侧已手动标注）
ax.set_yticks([])

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(False)

# 去掉图例框，改为在波形末端直接标注
ax.text(t_end + 0.02, V_GH, r"$V_{GS}$波形", color=C_VGS, fontsize=FONTSIZE_ANNO, fontweight="bold", va="center")
ax.text(t_end + 0.02, I_L,  r"$I_D$波形",    color=C_ID,  fontsize=FONTSIZE_ANNO, fontweight="bold", va="center")
ax.text(t_end + 0.02, V_DSon, r"$V_{DS}$波形", color=C_VDS, fontsize=FONTSIZE_ANNO, fontweight="bold", va="center")

plt.tight_layout()

# ── 保存 ──────────────────────────────────────────────────
out_dir = Path(__file__).parent.parent
svg_path = out_dir / "mosfet_turnon_waveform.svg"
png_path = out_dir / "mosfet_turnon_waveform.png"

fig.savefig(svg_path, format="svg", bbox_inches="tight")
fig.savefig(png_path, format="png", bbox_inches="tight", dpi=200)

print(f"[OK] SVG saved: {svg_path}")
print(f"[OK] PNG saved: {png_path}")
