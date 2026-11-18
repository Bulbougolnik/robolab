"""Визуализация: показывает, как работают SO3/SE3, Projection, MassMatrix и нога 4R.

Каждый рисунок строится вызовами библиотеки (FK, якобиан, M, g), а не отдельным кодом.
matplotlib подключается только здесь: математическое ядро его не импортирует.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from . import dynamics
from .models import LegModel, leg4r
from .operators import Projection
from .transforms import SE3, SO3, Point3, Vector3

C_AX = ("#d62728", "#2ca02c", "#1f77b4")          # оси x, y, z


# ======================================================================
# Вспомогательное
# ======================================================================
def _pts(model: LegModel, q):
    fr = model.chain.frames(q)
    P = {k: v.p.copy() for k, v in fr.items()}
    P["toe"] = model.chain.point(q, model.markers["toe"]).xyz
    P["heel"] = model.chain.point(q, model.markers["heel"]).xyz
    return P, fr


def _hip_y(model):
    return model.chain.joints[0].p0[1]


def _leg_com(model, q):
    c = np.array(dynamics.com_world(model, q))
    m = np.array([b.mass for b in model.bodies])
    return (m[:, None] * c).sum(0) / m.sum()


def _stand_shift(model, q):
    """Сдвиг, при котором голеностоп стоит в x=0, а подошва лежит на земле z=0."""
    P, _ = _pts(model, q)
    return np.array([-P["ankle"][0], 0.0, -min(P["toe"][2], P["heel"][2])])


def _arrow(ax, a, b, color="k", lw=2, style="->", ls="-"):
    ax.annotate("", xy=b, xytext=a,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw, ls=ls, shrinkA=0, shrinkB=0))


def _frame2d(ax, T: SE3, scale=0.5, name="", alpha=1.0):
    o, R = T.p[:2], T.R.matrix
    for i in (0, 1):
        _arrow(ax, o, o + scale * R[:2, i], C_AX[i], lw=2)
        ax.plot([], [])
    ax.text(*(o + [0.05, -0.18]), name, fontsize=10, alpha=alpha)


# ======================================================================
# 1. Точка и вектор (слайды 9, 19)
# ======================================================================
def demo_point_vs_vector():
    """Один и тот же ввод (1,0): точка получает перенос, вектор — нет."""
    fig, axs = plt.subplots(1, 2, figsize=(11, 5))
    pB, vB = Point3([1, 0, 0], "B"), Vector3([1, 0, 0], "B")
    W = SE3.identity("W")

    ax = axs[0]
    T = SE3(SO3.Rz(np.pi / 2), [1, 2, 0], "W", "B")
    pW, vW = T.apply_point(pB), T.apply_vector(vB)
    _frame2d(ax, W, 0.6, "W"); _frame2d(ax, T, 0.6, "B")
    _arrow(ax, (0, 0), pW.xyz[:2], "tab:orange", 2.5)
    ax.plot(*pW.xyz[:2], "o", color="tab:orange", ms=10)
    ax.text(pW.xyz[0] + .1, pW.xyz[1], f"точка: RP+p = {np.round(pW.xyz[:2], 2)}", color="tab:orange")
    _arrow(ax, (0, 0), vW.xyz[:2], "tab:purple", 3)
    ax.text(.1, .45, f"вектор: Rv = {np.round(vW.xyz[:2], 2)}", color="tab:purple")
    ax.set_title("R = Rz(90°), p = (1, 2): вход (1, 0) в системе B")

    ax = axs[1]
    for p, col, mk in (([1, 2, 0], "tab:blue", "o"), ([3, -1, 0], "tab:red", "s")):
        T = SE3(SO3.Rz(np.pi / 2), p, "W", "B")
        pW, vW = T.apply_point(pB), T.apply_vector(vB)
        ax.plot(*np.array(p)[:2], "x", color=col, ms=9)
        ax.plot(*pW.xyz[:2], mk, color=col, ms=10, label=f"точка при p={p[:2]} → {np.round(pW.xyz[:2], 2)}")
        _arrow(ax, np.array(p)[:2], np.array(p)[:2] + vW.xyz[:2], col, 2.5, ls="--")
    ax.set_title("Переносим начало B: точка уехала, направление вектора то же (0, 1)")
    ax.legend(loc="lower left", fontsize=8)
    for ax in axs:
        ax.set_aspect("equal"); ax.grid(alpha=.3); ax.set_xlim(-1, 4.2); ax.set_ylim(-1.6, 3.8)
        ax.set_xlabel("x в W, м"); ax.set_ylabel("y в W, м")
    fig.tight_layout()
    return fig


# ======================================================================
# 2. Порядок композиции (слайды 20, 21)
# ======================================================================
def demo_composition_order():
    """Rz(90°)·Tx(0.4) и Tx(0.4)·Rz(90°): разный результат."""
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.6))
    Rz = SO3.Rz(np.pi / 2)
    cases = [
        ("Сначала поворот родителя, потом сдвиг по ЕГО x",
         SE3(Rz, [0, 0, 0], "W", "B"), SE3(SO3.identity(), [.4, 0, 0], "B", "S")),
        ("Сначала сдвиг, потом поворот",
         SE3(SO3.identity(), [.4, 0, 0], "W", "B"), SE3(Rz, [0, 0, 0], "B", "S")),
    ]
    for ax, (title, T1, T2) in zip(axs, cases):
        T = T1 @ T2                                   # проверка имён внутри @
        _frame2d(ax, SE3.identity("W"), .3, "W")
        _frame2d(ax, T1, .22, "B", alpha=.6)
        _frame2d(ax, T, .3, "S")
        ax.plot([0, T1.p[0], T.p[0]], [0, T1.p[1], T.p[1]], "k:", lw=1)
        ax.set_title(f"{title}\np = {np.round(T.p, 2)}", fontsize=10)
        ax.set_aspect("equal"); ax.grid(alpha=.3)
        ax.set_xlim(-.3, .8); ax.set_ylim(-.3, .8)
        ax.set_xlabel("x в W, м"); ax.set_ylabel("y в W, м")
    fig.tight_layout()
    return fig


# ======================================================================
# 3. Нога: 3D, вид сбоку и спереди
# ======================================================================
def draw_frame3d(ax, T: SE3, scale=0.08, name=None):
    for i in range(3):
        d = T.R.matrix[:, i] * scale
        ax.quiver(*T.p, *d, color=C_AX[i], lw=1.6, arrow_length_ratio=0.3)
    if name:
        ax.text(*(T.p + [0.02, 0.02, 0.02]), name, fontsize=8)


def plot_leg_3d(model: LegModel, q, ax=None, frames=True):
    if ax is None:
        ax = plt.figure(figsize=(5.5, 5.5)).add_subplot(projection="3d")
    P, fr = _pts(model, q)
    hy = _hip_y(model)
    bar = np.array([[0, -hy, 0], [0, hy, 0]])
    ax.plot(*bar.T, color="gray", lw=6)
    ax.plot(*np.array([P["hip_pitch"], P["knee"], P["ankle"]]).T, "-o", color="k", lw=3)
    ax.plot(*np.array([P["ankle"], P["toe"], P["heel"], P["ankle"]]).T, color="tab:brown", lw=3)
    com = np.array(dynamics.com_world(model, q))
    m = np.array([b.mass for b in model.bodies])
    ax.scatter(*com.T, s=40 * m, color="tab:orange", depthshade=False)
    lc = _leg_com(model, q)
    zg = min(P["toe"][2], P["heel"][2])
    ax.scatter(*lc, marker="*", s=200, color="red", depthshade=False)
    ax.plot([lc[0]] * 2, [lc[1]] * 2, [lc[2], zg], "r--", lw=1)
    X, Y = np.meshgrid([-.45, .45], [-.3, .5])
    ax.plot_surface(X, Y, np.full_like(X, zg), alpha=.15, color="gray")
    if frames:
        for n in ("pelvis", "hip_pitch", "knee", "ankle"):
            draw_frame3d(ax, fr[n], 0.09, n)
    ax.set_xlim(-.55, .55); ax.set_ylim(-.4, .7); ax.set_zlim(-.95, .15)
    ax.set_box_aspect((1.1, 1.1, 1.1))
    ax.set_xlabel("x вперёд"); ax.set_ylabel("y влево"); ax.set_zlabel("z вверх")
    ax.view_init(elev=14, azim=-58)
    ax.set_title("3D: красная/зелёная/синяя = оси x/y/z систем звеньев", fontsize=9)
    return ax


def plot_leg_2d(ax, model: LegModel, q, plane="sagittal", shift=(0, 0, 0), show_com=True):
    dims = [0, 2] if plane == "sagittal" else [1, 2]
    s = np.asarray(shift, float)
    P, _ = _pts(model, q)
    g = lambda v: (np.asarray(v) + s)[dims]
    hy = _hip_y(model)
    bar = np.array([[0, -hy, 0], [0, hy, 0]]) + s
    ax.plot(bar[:, dims[0]], bar[:, dims[1]], color="gray", lw=7, solid_capstyle="round")
    seg = np.array([g(P["hip_pitch"]), g(P["knee"]), g(P["ankle"])])
    ax.plot(seg[:, 0], seg[:, 1], "-o", color="k", lw=3)
    foot = np.array([g(P["ankle"]), g(P["toe"]), g(P["heel"]), g(P["ankle"])])
    ax.plot(foot[:, 0], foot[:, 1], color="tab:brown", lw=3)
    zg = min(P["toe"][2], P["heel"][2]) + s[2]
    ax.axhline(zg, color="0.6", lw=1)
    if show_com:
        com = np.array(dynamics.com_world(model, q))
        m = np.array([b.mass for b in model.bodies])
        c2 = np.array([g(c) for c in com])
        ax.scatter(c2[:, 0], c2[:, 1], s=40 * m, color="tab:orange", zorder=3)
        lc = g(_leg_com(model, q))
        ax.plot(*lc, "*", color="red", ms=15, zorder=4)
        ax.plot([lc[0], lc[0]], [lc[1], zg], "r--", lw=1)
        if plane == "sagittal":
            lo, hi = sorted([g(P["heel"])[0], g(P["toe"])[0]])
            ok = lo <= lc[0] <= hi
            ax.plot([lo, hi], [zg, zg], lw=6, color="tab:green" if ok else "tab:red", alpha=.6)
            ax.set_title("вид сбоку: " + ("COM ноги над стопой" if ok else "COM ноги ВНЕ стопы"), fontsize=9)
    if plane != "sagittal":
        ax.set_title("вид спереди", fontsize=9)
    ax.set_aspect("equal"); ax.grid(alpha=.3)
    ax.set_xlabel("x вперёд, м" if plane == "sagittal" else "y влево, м")
    ax.set_ylabel("z вверх, м")


def show_leg(model: LegModel | None = None, q=(0, -.3, .6, -.3), frames=True):
    """Три вида одной позы + подпись с q."""
    model = model or leg4r()
    q = np.asarray(q, float)
    fig = plt.figure(figsize=(14, 4.8))
    plot_leg_3d(model, q, fig.add_subplot(1, 3, 1, projection="3d"), frames)
    plot_leg_2d(fig.add_subplot(1, 3, 2), model, q, "sagittal")
    plot_leg_2d(fig.add_subplot(1, 3, 3), model, q, "frontal")
    fig.suptitle("q = [крен, тангаж бедра, колено, голеностоп] = " + str(np.round(q, 2)) + " рад")
    fig.tight_layout()
    return fig


def interactive_leg(model: LegModel | None = None):
    """Ползунки суставов (Colab / Jupyter): видно, как q меняет FK и COM."""
    import ipywidgets as w
    model = model or leg4r()

    def draw(hip_roll=0.0, hip_pitch=-0.3, knee=0.6, ankle=-0.3):
        fig = show_leg(model, [hip_roll, hip_pitch, knee, ankle])
        plt.show()
        plt.close(fig)

    sl = lambda lo, hi, v: w.FloatSlider(value=v, min=lo, max=hi, step=0.05, continuous_update=False)
    w.interact(draw, hip_roll=sl(-.6, .6, 0), hip_pitch=sl(-1.2, 1.2, -.3),
               knee=sl(0, 2.2, .6), ankle=sl(-1, 1, -.3))


# ======================================================================
# 4. Проектор: реакция опоры (слайд 10)
# ======================================================================
def plot_force_projection(slope_deg=15.0, F=(80.0, 0.0, 300.0), mu=0.4):
    """F = F_n + F_t с помощью Projection.from_normal; конус трения |F_t| <= mu |F_n|."""
    a = np.deg2rad(slope_deg)
    n = np.array([-np.sin(a), 0, np.cos(a)])
    t = np.array([np.cos(a), 0, np.sin(a)])
    P = Projection.from_normal(n)
    F = np.asarray(F, float)
    Fn, Ft = P.split(F)
    ok = np.linalg.norm(Ft) <= mu * np.linalg.norm(Fn)
    k = 0.004
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot([-.6 * t[0], .6 * t[0]], [-.6 * t[2], .6 * t[2]], color="gray", lw=5)
    for s in (+1, -1):
        ang = np.arctan(mu) * s
        c, sn = np.cos(ang), np.sin(ang)
        d = np.array([c * n[0] + sn * n[2], c * n[2] - sn * n[0]])
        ax.plot([0, .6 * d[0]], [0, .6 * d[1]], "--", color="tab:green", lw=1)
    _arrow(ax, (0, 0), k * F[[0, 2]], "k", 3)
    _arrow(ax, (0, 0), k * Fn[[0, 2]], "tab:blue", 2.5)
    _arrow(ax, k * Fn[[0, 2]], k * F[[0, 2]], "tab:red", 2.5)
    ax.text(*(k * F[[0, 2]] + [.02, .02]), f"F, |F|={np.linalg.norm(F):.0f} Н")
    ax.text(*(k * Fn[[0, 2]] + [-.35, 0]), f"F_n=Pf\n{np.linalg.norm(Fn):.0f} Н", color="tab:blue")
    ax.text(*(k * F[[0, 2]] * 0.5 + k * Ft[[0, 2]] * .5 + [.02, -.08]),
            f"F_t=(I−P)F\n{np.linalg.norm(Ft):.0f} Н", color="tab:red")
    ax.set_title(f"Наклон {slope_deg:.0f}°, μ={mu}: " + ("скольжения нет" if ok else "НОГА СКОЛЬЗИТ")
                 + f"\n|F_t|/|F_n| = {np.linalg.norm(Ft) / np.linalg.norm(Fn):.2f}")
    ax.set_aspect("equal"); ax.grid(alpha=.3); ax.set_xlim(-.7, .9); ax.set_ylim(-.3, 1.5)
    ax.set_xlabel("x, м (масштаб силы условный)"); ax.set_ylabel("z, м")
    return fig


# ======================================================================
# 5. Матрица инерции (MassMatrix)
# ======================================================================
def plot_mass_matrix(model: LegModel | None = None, qs=None, labels=None, idx=(1, 2)):
    """M(q) для разных поз и эллипс «T = const» для пары суставов idx."""
    model = model or leg4r()
    qs = qs or [np.array([0, 0, 0, 0.]), np.array([0, -.8, 1.6, -.8])]
    labels = labels or ["нога прямая", "присед"]
    Ms = [dynamics.mass_matrix(model, q) for q in qs]
    vmax = max(M.matrix.max() for M in Ms)
    fig, axs = plt.subplots(1, len(qs) + 1, figsize=(4.6 * (len(qs) + 1), 4.4))
    names = ["roll", "hip", "knee", "ankle"]
    for ax, M, lb in zip(axs, Ms, labels):
        im = ax.imshow(M.matrix, cmap="viridis", vmin=0, vmax=vmax)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{M.matrix[i, j]:.2f}", ha="center", va="center", color="w", fontsize=8)
        ax.set_xticks(range(4)); ax.set_xticklabels(names); ax.set_yticks(range(4)); ax.set_yticklabels(names)
        ax.set_title(f"M(q): {lb}\nλ_min={np.linalg.eigvalsh(M.matrix).min():.3f} > 0", fontsize=9)
    ax = axs[-1]
    th = np.linspace(0, 2 * np.pi, 200)
    u = np.vstack([np.cos(th), np.sin(th)])
    for M, lb in zip(Ms, labels):
        Msub = M.matrix[np.ix_(idx, idx)]
        r = np.sqrt(2.0 / np.einsum("in,ij,jn->n", u, Msub, u))      # 1/2 dq^T M dq = 1 Дж
        ax.plot(r * u[0], r * u[1], label=lb, lw=2)
    ax.set_aspect("equal"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    ax.set_xlabel(f"dq {names[idx[0]]}, рад/с"); ax.set_ylabel(f"dq {names[idx[1]]}, рад/с")
    ax.set_title("Скорости с одинаковой энергией 1 Дж:\nчем уже эллипс, тем «тяжелее» направление", fontsize=9)
    fig.tight_layout()
    return fig


# ======================================================================
# 6. Карты энергии и гравитационных моментов
# ======================================================================
def plot_gravity_maps(model: LegModel | None = None, n=31):
    """U и g(q) на сетке (тангаж бедра × колено), стопа параллельна полу."""
    model = model or leg4r()
    hips = np.linspace(-1.2, 0.4, n)
    knees = np.linspace(0.0, 2.2, n)
    U = np.zeros((n, n)); G = np.zeros((2, n, n))
    for i, k in enumerate(knees):
        for j, h in enumerate(hips):
            q = np.array([0, h, k, -(h + k)])
            U[i, j] = dynamics.potential_energy(model, q)
            g = dynamics.gravity_vector(model, q)
            G[0, i, j], G[1, i, j] = g[1], g[2]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, Z, t in zip(axs, (U, G[0], G[1]),
                        ("U(q), Дж (отн. таза)", "g(q) бедро, Н·м", "g(q) колено, Н·м")):
        cs = ax.contourf(hips, knees, Z, 20, cmap="coolwarm")
        fig.colorbar(cs, ax=ax); ax.set_title(t)
        ax.set_xlabel("тангаж бедра, рад"); ax.set_ylabel("колено, рад")
    fig.suptitle("Стопа параллельна полу: ankle = −(hip + knee)")
    fig.tight_layout()
    return fig


# ======================================================================
# 7. Анимация приседа: FK + якобиан + g(q) вместе
# ======================================================================
def animate_squat(model: LegModel | None = None, frames=60, k_max=1.7):
    """Возвращает FuncAnimation. В Colab: HTML(anim.to_jshtml())."""
    from matplotlib.animation import FuncAnimation
    model = model or leg4r()
    half = frames // 2
    ks = np.r_[np.linspace(0.05, k_max, half), np.linspace(k_max, 0.05, half)]
    qs = [np.array([0, -k / 2, k, -k / 2]) for k in ks]
    tau = np.array([dynamics.gravity_vector(model, q)[1:] for q in qs])

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.8), gridspec_kw={"width_ratios": [1, 1.2]})
    for c, lb in enumerate(("бедро", "колено", "голеностоп")):
        bx.plot(ks if False else np.arange(len(ks)), tau[:, c], label=lb)
    bx.set_xlabel("кадр"); bx.set_ylabel("g(q), Н·м"); bx.grid(alpha=.3); bx.legend()
    bx.set_title("Моменты, удерживающие ногу против гравитации")
    cursor = bx.axvline(0, color="k", lw=1)

    def update(i):
        ax.clear()
        plot_leg_2d(ax, model, qs[i], "sagittal", shift=_stand_shift(model, qs[i]))
        ax.set_xlim(-.7, .7); ax.set_ylim(-.1, 1.0)
        cursor.set_xdata([i, i])
        return ()

    anim = FuncAnimation(fig, update, frames=len(qs), interval=60, blit=False)
    plt.close(fig)
    return anim


def show_animation(anim):
    from IPython.display import HTML, display
    display(HTML(anim.to_jshtml()))
