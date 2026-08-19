"""Figures for RESULTS.md.

Chrome (ink, grid, surface) comes from the repo's `palette.py` so these sit
alongside the existing figures. Series identity uses the validated categorical
order (blue, orange, aqua, yellow) — checked with the dataviz validator against
this surface: adjacent CVD dE 9.1, normal-vision dE 22.9, all pass; aqua and
yellow fall below 3:1 contrast, so every series is DIRECT-LABELLED as well as
legended (the relief rule) and identity is never carried by colour alone.
Single (light) mode on purpose: these are static PNGs in a repo document.
"""

import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, '/workspace/tensor_language')
from palette import INK, SECONDARY, MUTED, GRID, SURFACE

SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
FIG = '/workspace/tensor_language/basis_aligned/bilinear_quotient/figures'
plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
    'axes.edgecolor': GRID, 'axes.labelcolor': SECONDARY, 'text.color': INK,
    'xtick.color': SECONDARY, 'ytick.color': SECONDARY,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
    'font.size': 9, 'axes.titlesize': 10, 'legend.frameon': False,
    'axes.spines.top': False, 'axes.spines.right': False,
})


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, color=INK, loc='left', fontweight='bold', pad=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_axisbelow(True)


def label_end(ax, x, y, text, color, dy=0, va='center'):
    ax.annotate(text, (x, y), xytext=(5, dy), textcoords='offset points',
                color=color, fontsize=8, va=va, ha='left', fontweight='bold')


# ------------------------------------------------------- fig 1: A2 dynamics
def fig_dynamics():
    d = json.load(open(f'{FIG}/../a2_extra_results.json'))['dynamics_controlled'][1:]
    step = np.array([r['step'] for r in d])          # step 0 dropped: log axis
    series = [('Fourier-block projection', 'fourier_proj_test_acc', SERIES[0], 'Fourier projection'),
              ('the model itself', 'test_acc', SERIES[1], 'the model'),
              ('control: relabelled-group Fourier', 'ctrl_relabelled_fourier_acc', SERIES[3], 'relabelled ctrl'),
              ('control: random 4-dim blocks', 'ctrl_random_blocks_acc', SERIES[2], 'random ctrl')]
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    for si, (name, key, c, short) in enumerate(series):
        v = np.array([r[key] for r in d])
        ax.plot(step, v, lw=2, color=c, label=name, solid_capstyle='round')
        label_end(ax, step[-1], v[-1], short, c, dy=(-11 if si == 1 else 0))
    ax.set_xscale('log')
    ax.set_ylim(-0.03, 1.10)
    ax.set_xlim(4e2, 1.6e5)
    ax.set_xticks([1e3, 1e4, 4e4])
    ax.set_xticklabels(['1k', '10k', '40k'])
    style(ax, 'From step 1500 the clean part of the weights already RANKS every held-out\npair correctly, while the model itself scores zero',
          'training step (log)', 'test accuracy')
    ax.axvline(1500, color=MUTED, lw=1, ls=(0, (3, 3)))
    ax.annotate('ranking correct from here', (1500, 0.52), xytext=(6, 0),
                textcoords='offset points', color=MUTED, fontsize=8, rotation=90, va='center')
    ax.legend(fontsize=8, labelcolor=SECONDARY, ncol=4, loc='upper center',
              bbox_to_anchor=(0.5, -0.19))
    fig.tight_layout()
    fig.savefig(f'{FIG}/a2_dynamics.png', dpi=170)
    plt.close(fig)


# ------------------------------------------- fig 2: A2 instrument calibration
def fig_calibration():
    cal = json.load(open(f'{FIG}/../a2_jade_results.json'))
    j_off = [r['off_block'] for r in cal['calibration']]
    j_rec = [r['best_n_freq_full'] for r in cal['calibration']]
    # commutant routes, from a2_calibration.json (isotropic sweep)
    c2 = json.load(open(f'{FIG}/../a2_calibration.json'))
    c_off = [r['off_block_mass'] for r in c2['noise_iso']]
    exact = [r['exact']['n_freq_recovered'] for r in c2['noise_iso']]
    robust = [r['robust_hinted']['n_freq_recovered'] for r in c2['noise_iso']]
    tr = [(r['off_block'], r['best_n_freq_full']) for r in cal['trained']]

    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    eps = 3e-4
    for x, y, name, c in ((c_off, exact, 'exact SBD (commutant)', SERIES[3]),
                          (c_off, robust, 'approximate commutant', SERIES[1]),
                          (j_off, j_rec, 'JADE + stratification tolerance', SERIES[0])):
        x = [max(v, eps) for v in x]
        ax.plot(x, y, lw=2, marker='o', ms=5, color=c, label=name,
                markerfacecolor=SURFACE, markeredgewidth=1.6, solid_capstyle='round')
    # the model's OWN residual, scaled by alpha: the only curve made of real material
    fine = json.load(open(f'{FIG}/../a2_residual_scale_fine.json'))
    for si, (seed, rows) in enumerate(sorted(fine.items())):
        ax.plot([r['off_block'] for r in rows], [r['best'] for r in rows], lw=1.6,
                marker='D', ms=4.5, color=SERIES[2], alpha=0.85,
                markerfacecolor=SERIES[2], markeredgecolor=SURFACE, markeredgewidth=1.2,
                label='trained model, own residual scaled (3 seeds)' if si == 0 else None)
    ax.scatter([t[0] for t in tr], [t[1] for t in tr], s=70, marker='D', zorder=6,
               color=SERIES[2], edgecolor=INK, linewidth=1.2)
    ax.annotate('the trained models\n(full residual)', (tr[1][0], tr[1][1]), xytext=(8, -14),
                textcoords='offset points', fontsize=8, color=SERIES[2], fontweight='bold')
    ax.set_xscale('log')
    ax.set_xlim(eps * 0.7, 0.7)
    ax.set_ylim(-0.5, 12)
    ax.set_yticks(range(0, 12, 2))
    style(ax, 'Block recovery: instrument ceilings on known truth, and where the real residual sits',
          'off-block mass of the family (log)', 'frequencies fully recovered (of 11)')
    ax.legend(loc='lower left', fontsize=8, labelcolor=SECONDARY)
    fig.tight_layout()
    fig.savefig(f'{FIG}/a2_calibration.png', dpi=170)
    plt.close(fig)


# ------------------------------------------------------- fig 3: A4 band axis
def fig_band_axis():
    d = json.load(open(f'{FIG}/../a4_results.json'))
    pc = d['per_component']
    gammas = sorted({r['gamma'] for r in pc}, reverse=True)
    rhos = sorted({r['rho'] for r in pc})
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.8), sharey=True)

    def panel(ax, xfac, dodgefac, xlabels, title, xlabel):
        for i, xv in enumerate(sorted({r[xfac] for r in pc}, reverse=(xfac == 'gamma'))):
            for k, dv in enumerate(sorted({r[dodgefac] for r in pc})):
                pts = [r for r in pc if r[xfac] == xv and r[dodgefac] == dv]
                gi = [gammas.index(r['gamma']) for r in pts]
                xs = [i + (k - 1) * 0.22 + (n - 0.5) * 0.07 for n in range(len(pts))]
                ys = [max(r['curvature_ratio'], 5e-5) for r in pts]
                ax.scatter(xs, ys, s=44, color=[SERIES[g] for g in gi],
                           edgecolor=SURFACE, linewidth=1.3, zorder=4)
        ax.set_xticks(range(len(xlabels)))
        ax.set_xticklabels(xlabels)
        ax.set_xlim(-0.6, len(xlabels) - 0.4)
        ax.set_yscale('log')
        ax.axhline(1.0, color=MUTED, lw=1, ls=(0, (3, 3)))
        style(ax, title, xlabel, '')

    panel(axes[0], 'gamma', 'rho', ['1.0', '0.1', '0.01'],
          "grouped by gain γ  —  r = +0.00", "planted gain  γ   (the doc's predicted axis)")
    panel(axes[1], 'rho', 'gamma', ['0', '2', '10'],
          'grouped by curvature ρ  —  r = −0.99', 'measured mean-to-fluctuation  ρ')
    axes[0].set_ylabel('linearize error / prune error')
    axes[0].annotate('linearizing = pruning', (-0.5, 1.0), xytext=(0, 5),
                     textcoords='offset points', color=MUTED, fontsize=8)
    handles = [plt.Line2D([], [], marker='o', ls='', color=SERIES[i], label=f'γ = {g}',
                          markersize=7, markeredgecolor=SURFACE)
               for i, g in enumerate(gammas)]
    axes[1].legend(handles=handles, fontsize=8, labelcolor=SECONDARY, loc='center right')
    fig.suptitle('Whether a component can be linearized is set by curvature, not by gain',
                 color=INK, fontweight='bold', x=0.012, ha='left', y=0.99, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(f'{FIG}/a4_band_axis.png', dpi=170)
    plt.close(fig)


# -------------------------------------------------------- fig 4: A4 frontier
def fig_frontier():
    d = json.load(open(f'{FIG}/../a4_results.json'))['frontier']
    kp = [(r['budget'], r['err']) for r in d['keep_prune'] if r['err'] == r['err']]
    al = [(r['budget'], r['err']) for r in d['keep_prune_linearize'] if r['err'] == r['err']]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for pts, name, c in ((kp, 'keep / prune', SERIES[1]), (al, 'keep / prune / linearize', SERIES[0])):
        x = [p[0] for p in pts]
        y = [max(p[1], 1e-9) for p in pts]
        ax.plot(x, y, lw=2, marker='o', ms=4, color=c, label=name,
                markerfacecolor=SURFACE, markeredgewidth=1.4, solid_capstyle='round')
    ax.set_yscale('log')
    style(ax, 'Linearizing the middle band dominates pruning at 22 of 25 parameter budgets',
          'parameters retained', 'functional error (measured)')
    ax.legend(loc='upper right', fontsize=8, labelcolor=SECONDARY)
    fig.tight_layout()
    fig.savefig(f'{FIG}/a4_frontier.png', dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    fig_dynamics()
    fig_calibration()
    fig_band_axis()
    fig_frontier()
    print('wrote 4 figures to', FIG)
