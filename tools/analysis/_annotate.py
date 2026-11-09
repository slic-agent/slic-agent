"""Value labels for curve figures.

`annotate_values` writes one label per curve at a given x (the curve's value there, in the curve's colour), stacked so labels
never overlap: labels are pushed apart vertically to `min_gap` (data units) while keeping their order, and a thin leader joins a
label to its point whenever it had to move. `mark_x` draws the dotted vertical line that names a reference setting (e.g. the
8-turn budget or the 5 % tolerance). The numbers come from the same arrays the figure draws; nothing is recomputed here.
"""


def stack_positions(ys, min_gap, lo, hi):
    """Adjusted y for each label so consecutive labels (sorted by value) are >= min_gap apart and all lie within [lo, hi]."""
    ys = [float(v) for v in ys]
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    pos = [None] * len(ys)
    last = lo - min_gap
    for i in order:                       # bottom-up: push up
        pos[i] = max(ys[i], last + min_gap); last = pos[i]
    nxt = hi + min_gap
    for i in reversed(order):             # top-down: pull back under the ceiling, keeping the gaps
        pos[i] = min(pos[i], nxt - min_gap); nxt = pos[i]
    return pos


def annotate_values(ax, x, items, side, fmt="{:.1f}%", fontsize=8.5, min_gap=3.4, xpad=0.35, ylim=None, zorder=6, yoff=0.0):
    """items: list of (y, colour). side 'left' puts the label left of x (right-aligned), 'right' puts it right of x.
    yoff shifts every label by that many data units (a label inside the panel can go under its curve, joined by a leader)."""
    if not items:
        return
    lo, hi = ylim if ylim is not None else ax.get_ylim()
    ys = [y for y, _ in items]
    pos = stack_positions([y + yoff for y in ys], min_gap, lo + 0.6 * min_gap, hi - 0.6 * min_gap)
    for (y, colour), yl in zip(items, pos):
        xt = x - xpad if side == "left" else x + xpad
        moved = abs(yl - y) > 0.25 * min_gap
        ax.annotate(fmt.format(y), xy=(x, y), xytext=(xt, yl), textcoords="data", fontsize=fontsize, color=colour,
                    ha="right" if side == "left" else "left", va="center", zorder=zorder, annotation_clip=False,
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85),
                    arrowprops=dict(arrowstyle="-", color=colour, lw=0.6, alpha=0.7, shrinkA=0, shrinkB=1.5) if moved else None)


def mark_x(ax, x, text, y_text=None, fontsize=8, color="#6b6660", dx=0.25):
    """Dotted vertical line at x with a small label at the top of the axes (to the right of the line)."""
    ax.axvline(x, color="gray", ls=":", lw=0.9, alpha=0.8, zorder=2)
    lo, hi = ax.get_ylim()
    ax.text(x + dx, hi - 0.015 * (hi - lo) if y_text is None else y_text, text, fontsize=fontsize, color=color, va="top", ha="left",
            zorder=6, bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))
