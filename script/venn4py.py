# SPDX-FileCopyrightText: (c) UIUC PurpCode Team
#
# SPDX-License-Identifier: Apache-2.0

from itertools import combinations

import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse


# get shared elements for each combination of sets
def get_shared(sets):
    IDs = sets.keys()
    combs = sum(
        [list(map(list, combinations(IDs, i))) for i in range(1, len(IDs) + 1)], []
    )

    shared = {}
    for comb in combs:
        ID = " and ".join(comb)
        if len(comb) == 1:
            shared.update({ID: sets[comb[0]]})
        else:
            setlist = [sets[c] for c in comb]
            u = set.intersection(*setlist)
            shared.update({ID: u})
    return shared


# get unique elements for each combination of sets
def get_unique(shared):
    unique = {}
    for shar in shared:
        if shar == list(shared.keys())[-1]:
            s = shared[shar]
            unique.update({shar: s})
            continue
        count = shar.count(" and ")
        if count == 0:
            setlist = [
                shared[k] for k in shared.keys() if k != shar and " and " not in k
            ]
            s = shared[shar].difference(*setlist)
        else:
            setlist = [
                shared[k]
                for k in shared.keys()
                if k != shar and k.count(" and ") >= count
            ]
            s = shared[shar].difference(*setlist)
        unique.update({shar: s})
    return unique


# plot Venn
def venny4py(
    sets,
    ax,
    size=3.5,
    colors="bgrc",
    line_width=None,
    font_size=None,
    legend_cols=2,
    column_spacing=4,
):
    assert len(sets) == 4, "Number of sets must be 4"
    shared = get_shared(sets)
    unique = get_unique(shared)
    ce = colors
    lw = size * 0.5 if line_width is None else line_width
    fs = size * 2 if font_size is None else font_size
    nc = legend_cols
    cs = column_spacing

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # draw ellipses
    ew = 45  # width
    eh = 75  # height
    xe = [35, 48, 52, 65]  # x coordinates
    ye = [35, 45, 45, 35]  # y coordinates
    ae = [225, 225, 315, 315]  # angles

    for i, s in enumerate(sets):
        ax.add_artist(
            Ellipse(xy=(xe[i], ye[i]), width=ew, height=eh, fc=ce[i], angle=ae[i])
        )
        ax.add_artist(
            Ellipse(
                xy=(xe[i], ye[i]),
                width=ew,
                height=eh,
                fc="None",
                angle=ae[i],
                ec="royalblue" if i == 0 else None,
                lw=lw,
            )
        )

    # annotate
    xt = [
        10,
        32,
        68,
        91,
        14,
        34,
        66,
        86,
        26,
        28,
        50,
        50,
        72,
        74,
        37,
        60,
        40,
        63,
        50,
    ]  # x
    yt = [
        67,
        79,
        79,
        67,
        41,
        70,
        70,
        41,
        59,
        26,
        11,
        60,
        26,
        59,
        51,
        17,
        17,
        51,
        35,
    ]  # y

    for j, s in enumerate(sets):
        ax.text(
            xt[j],
            yt[j],
            len(sets[s]),
            ha="center",
            va="center",
            fontsize=fs,
            transform=ax.transData,
        )

    for k in unique:
        j += 1
        ax.text(
            xt[j],
            yt[j],
            len(unique[k]),
            ha="center",
            va="center",
            fontsize=fs,
            transform=ax.transData,
        )

    # legend
    handles = [
        mpatches.Patch(color=ce[i], label=l, lw=lw, ec="royalblue" if i == 0 else None)
        for i, l in enumerate(sets)
    ]
    ax.legend(
        labels=sets,
        handles=handles,
        fontsize=fs,
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
        bbox_transform=ax.transAxes,
        loc=9,
        handlelength=1.5,
        ncol=nc,
        columnspacing=cs,
        handletextpad=0.5,
    )
