import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib

matplotlib.use("tkagg")

AREA = 100
N_AC = 6
N_AC_PTCS = 1000  # particles per aircraft

DECAY_SIGMA = 10.0
CONF_BOUND = (0.0, 1.0)  # confident normlization bound

NEW_PTC_D_SIGMA = 0.25  # of AREA size
PTC_WEIGHT_LENGTH_SIGMA = 0.5  # of AREA size
PTC_HDG_VARY_SIGMA = 0.01  # of AREA size
PTC_WALK_SIGMA = 0.01  # of AREA size

# aicraft
AC_X = np.array([])
AC_Y = np.array([])
AC_WX = np.array([])
AC_WY = np.array([])

# particles
PTC_X = np.array([])  # current position of particles
PTC_Y = np.array([])
PTC_WX = np.array([])  # particles' wind state
PTC_WY = np.array([])
PTC_AGE = np.array([])
PTC_X0 = np.array([])  # origin positions of particles
PTC_Y0 = np.array([])


def cleanup():
    mask = PTC_AGE < DECAY_SIGMA * 3
    mask &= PTC_X > 0
    mask &= PTC_X < AREA
    mask &= PTC_Y > 0
    mask &= PTC_Y < AREA
    return np.where(mask)[0]


def wind_grid():
    cd = AREA * PTC_WEIGHT_LENGTH_SIGMA
    ca = DECAY_SIGMA

    def strength(mask):
        """decaying factor of particles
        """
        ptc_ages = PTC_AGE[mask]

        strength = np.exp(-1 * ptc_ages ** 2 / (2 * ca ** 2))
        return strength

    def ptc_weights(x0, y0, mask):
        """particle weights are calculated as gaussian function
        of distances of particles to a grid point, particle age,
        and particle distance from its origin.
        """
        ptc_xs = PTC_X[mask]
        ptc_ys = PTC_Y[mask]

        ptc_x0s = PTC_X0[mask]
        ptc_y0s = PTC_Y0[mask]

        d = np.sqrt((ptc_xs - x0) ** 2 + (ptc_ys - y0) ** 2)
        fd = np.exp(-1 * d ** 2 / (2 * cd ** 2))

        d0s = np.sqrt((ptc_xs - ptc_x0s) ** 2 + (ptc_ys - ptc_y0s) ** 2)
        fd0 = np.exp(-1 * d0s ** 2 / (2 * cd ** 2))

        fa = strength(mask)

        weights = fd * fd0 * fa
        return weights

    def scaled_confidence(l):
        """kernel function to scale confidence values
        """
        a, b = CONF_BOUND
        l = np.array(l)
        lscale = (b - a) * (l - np.min(l)) / (np.max(l) - np.min(l)) + a
        return lscale

    xs = list(range(0, AREA + 10, 10))
    ys = list(range(0, AREA + 10, 10))
    xx, yy = np.meshgrid(xs, ys)
    coords_x = xx.flatten()
    coords_y = yy.flatten()
    coords_wwx = []
    coords_wwy = []
    coords_ptc_wei = []
    coords_ptc_num = []
    coords_ptc_hmg = []
    coords_ptc_str = []

    for x, y in zip(coords_x, coords_y):
        mask = (PTC_X > x - 5) & (PTC_X < x + 5) & (PTC_Y > y - 5) & (PTC_Y < y + 5)
        n = len(PTC_X[mask])
        if n > 0:
            ws = ptc_weights(x, y, mask)
            wx = np.sum(ws * PTC_WX[mask]) / np.sum(ws)
            wy = np.sum(ws * PTC_WY[mask]) / np.sum(ws)
            hmgs = np.linalg.norm(np.cov([PTC_WX[mask], PTC_WY[mask]]))
            strs = np.mean(strength(mask))
        else:
            ws = 0
            wx = 0
            wy = 0
            hmgs = 0
            strs = 0

        if np.isnan(hmgs):
            hmgs = 0

        coords_wwx.append(wx)
        coords_wwy.append(wy)

        coords_ptc_num.append(n)
        coords_ptc_wei.append(np.mean(ws))
        coords_ptc_hmg.append(hmgs)
        coords_ptc_str.append(strs)

    # compute confidence at each grid point, based on:
    #   particle numbers, mean weights, uniformness of particle headings
    fw = scaled_confidence(coords_ptc_wei)
    fn = scaled_confidence(coords_ptc_num)
    fh = scaled_confidence(coords_ptc_hmg)
    fs = scaled_confidence(coords_ptc_str)
    coords_conf = (fw + fn + fh + fs) / 4.0

    return (
        np.array(coords_x),
        np.array(coords_y),
        np.array(coords_wwx),
        np.array(coords_wwy),
        np.array(coords_conf),
    )


def plot_ac(ax, mask):
    for i, (x, y, wx, wy) in enumerate(zip(AC_X, AC_Y, AC_WX, AC_WY)):
        if mask[i]:
            color = "k"
        else:
            color = "r"

        ax.scatter(x, y, c=color)
        ax.arrow(x, y, wx, wy, lw=2, head_width=2, head_length=2, ec=color, fc=color)
        cir = plt.Circle(
            (x, y), radius=np.sqrt(AREA * 0.8), color=color, fc="none", ls="--", lw=2
        )
        ax.add_patch(cir)
    ax.set_xlim([0, AREA])
    ax.set_ylim([0, AREA])
    ax.set_aspect("equal")


def plot_particle_samples(ax, sample=10, draw_hdg=None):
    sortidx = np.argsort(PTC_AGE[::sample])[::-1]

    X = PTC_X[::sample][sortidx]
    Y = PTC_Y[::sample][sortidx]
    WX = PTC_WX[::sample][sortidx]
    WY = PTC_WY[::sample][sortidx]
    AGE = PTC_AGE[::sample][sortidx]

    if max(AGE) == min(AGE):
        Color = "gray"
    else:
        Color = cm.Blues(1 + 0.2 - (AGE - min(AGE)) / (max(AGE) - min(AGE)))

    ax.scatter(X, Y, s=3, color=Color)

    if draw_hdg:
        for i, (x, y, wx, wy) in enumerate(zip(X, Y, WX, WY)):
            ax.plot([x, x + wx / 2], [y, y + wy / 2], color="k", alpha=0.5, lw=1)

    ax.set_xlim([0, AREA])
    ax.set_ylim([0, AREA])
    ax.set_aspect("equal")


def plot_wind_grid(ax):
    x, y, wx, wy, conf = wind_grid()

    for x, y, wx, wy in zip(x, y, wx, wy):
        if wx != 0 and wy != 0:
            ax.scatter(x, y, s=15, color="k")
            ax.arrow(x, y, wx, wy, head_width=2, head_length=2, ec="k", fc="k")
        else:
            ax.scatter(x, y, s=15, color="k", facecolors="none")
    ax.set_aspect("equal")


def plot_wind_confidence(ax):
    x, y, wx, wy, conf = wind_grid()

    n = int(np.sqrt(len(x)))
    CS = ax.contourf(
        x.reshape(n, n),
        y.reshape(n, n),
        conf.reshape(n, n),
        levels=np.linspace(0, 1, 10),
        cmap=cm.get_cmap(cm.BuGn),
        alpha=0.8,
    )
    plt.colorbar(CS, fraction=0.046, pad=0.01)
    ax.set_aspect("equal")


def sample(n, error=False):
    # initialize aicraft
    x = np.random.normal(AREA / 2, AREA / 4, n)
    y = np.random.normal(AREA / 2, AREA / 4, n)

    v = np.random.normal(4, 1, n)
    hdg = np.random.normal(0.25 * np.pi, 0.1 * np.pi, n)

    if error:
        x[0] = AREA / 2
        y[0] = AREA / 2
        v[0] = np.random.normal(10, 0.5)
        hdg[0] = np.random.normal(0.25 * np.pi, 0.05 * np.pi)

    wx = v * np.sin(hdg)
    wy = v * np.cos(hdg)

    return x, y, wx, wy


def prob_ac_accept():
    if len(PTC_X) == 0:
        keep = [True] * len(AC_X)
    else:
        mu_wx = np.mean(PTC_WX)
        mu_wy = np.mean(PTC_WY)
        sigma_wx = np.std(PTC_WX) * 3
        sigma_wy = np.std(PTC_WY) * 3
        prob = np.exp(
            -0.5
            * (
                (AC_WX - mu_wx) ** 2 / (sigma_wx ** 2)
                + (AC_WY - mu_wy) ** 2 / (sigma_wy ** 2)
            )
        )
        choice = np.random.random(len(prob))
        print()
        print(prob)
        print(choice)
        keep = prob > choice

    return keep


fig = plt.figure(figsize=(15, 5))

for step in range(60):
    dt = 1

    # update aircraft
    if (step + 1) % 2 == 0:
        # each 2 time step, produce an error
        AC_X, AC_Y, AC_WX, AC_WY = sample(N_AC, error=True)
    else:
        AC_X, AC_Y, AC_WX, AC_WY = sample(N_AC)

    # update existing particles, random walk motion model
    n = len(PTC_X)
    if n > 0:
        PTC_X = PTC_X + np.random.normal(PTC_WX, PTC_WALK_SIGMA * AREA, n)
        PTC_Y = PTC_Y + np.random.normal(PTC_WY, PTC_WALK_SIGMA * AREA, n)
        PTC_AGE = PTC_AGE + 1

    # get global mean and variacne for error sample rejection
    mask = prob_ac_accept()

    # add new particles
    for x, y, wx, wy in zip(AC_X[mask], AC_Y[mask], AC_WX[mask], AC_WY[mask]):
        pxy = np.random.multivariate_normal(
            [x, y],
            [[NEW_PTC_D_SIGMA * AREA, 0], [0, NEW_PTC_D_SIGMA * AREA]],
            N_AC_PTCS,
        )

        px = pxy[:, 0]
        py = pxy[:, 1]
        pwx = wx * (1 + np.random.normal(0, PTC_HDG_VARY_SIGMA, N_AC_PTCS))
        pwy = wy * (1 + np.random.normal(0, PTC_HDG_VARY_SIGMA, N_AC_PTCS))

        PTC_X = np.append(PTC_X, px)
        PTC_Y = np.append(PTC_Y, py)
        PTC_WX = np.append(PTC_WX, pwx)
        PTC_WY = np.append(PTC_WY, pwy)
        PTC_AGE = np.append(PTC_AGE, np.zeros(N_AC_PTCS))
        PTC_X0 = np.append(PTC_X0, x * np.ones(N_AC_PTCS))
        PTC_Y0 = np.append(PTC_Y0, y * np.ones(N_AC_PTCS))

    # cleanup particle
    idx = cleanup()
    PTC_X = PTC_X[idx]
    PTC_Y = PTC_Y[idx]
    PTC_WX = PTC_WX[idx]
    PTC_WY = PTC_WY[idx]
    PTC_AGE = PTC_AGE[idx]
    PTC_X0 = PTC_X0[idx]
    PTC_Y0 = PTC_Y0[idx]

    # # when save image for gif
    # plt.close()
    # fig = plt.figure(figsize=(15, 5))

    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    plot_ac(ax1, mask)
    plot_particle_samples(ax2)
    plot_wind_confidence(ax3)
    plot_wind_grid(ax3)
    plt.tight_layout()

    # # convert -delay 30 -loop 0 *.png animation.gif
    # plt.savefig('/tmp/mp-sim-%02d.png' % (step+1))
    # plt.close()

    plt.draw()
    plt.waitforbuttonpress(-1)
    plt.clf()
