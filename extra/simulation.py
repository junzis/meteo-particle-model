import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

AREA = 100
N_AC = 5
N_AC_PTCS = 1000                # particles per aircraft

DECAY_SIGMA = 10.0
CONF_BOUND = (0.0, 1.0)         # confident normlization bound

NEW_PTC_D_SIGMA = 0.25          # of AREA size
PTC_WEIGHT_LENGTH_SIGMA = 0.5   # of AREA size
PTC_HDG_VARY_SIGMA = 0.01       # of AREA size
PTC_WALK_SIGMA = 0.01           # of AREA size

# aicraft
AC_X = np.array([])
AC_Y = np.array([])
AC_VX = np.array([])
AC_VY = np.array([])

# particles
PTC_X = np.array([])            # current position of particles
PTC_Y = np.array([])
PTC_WVX = np.array([])           # particles' wind state
PTC_WVY = np.array([])
PTC_AGE = np.array([])
PTC_X0 = np.array([])           # origin positions of particles
PTC_Y0 = np.array([])


def resample():
    mask = PTC_AGE < DECAY_SIGMA * 3
    mask &= PTC_X > 0
    mask &= PTC_X < AREA
    mask &= PTC_Y > 0
    mask &= PTC_Y < AREA
    return np.where(mask)[0]

def wind_grid():
    cd = AREA * PTC_WEIGHT_LENGTH_SIGMA
    ca = DECAY_SIGMA

    def strength( mask):
        """decaying factor of particles
        """
        ptc_ages = PTC_AGE[mask]

        strength = np.exp(-1 * ptc_ages**2 / (2 * ca**2))
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

        d = np.sqrt((ptc_xs-x0)**2 + (ptc_ys-y0)**2)
        fd = np.exp(-1 * d**2 / (2 * cd**2))

        d0s = np.sqrt((ptc_xs-ptc_x0s)**2 + (ptc_ys-ptc_y0s)**2)
        fd0 = np.exp(-1 * d0s**2 / (2 * cd**2))

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

    xs = list(range(0, AREA+10, 10))
    ys = list(range(0, AREA+10, 10))
    xx, yy = np.meshgrid(xs, ys)
    coords_x = xx.flatten()
    coords_y = yy.flatten()
    coords_wvx = []
    coords_wvy = []
    coords_ptc_wei = []
    coords_ptc_num = []
    coords_ptc_hmg = []
    coords_ptc_str = []

    for x, y in zip(coords_x, coords_y):
        mask = (PTC_X > x-5) & (PTC_X < x+5) & (PTC_Y > y-5) & (PTC_Y < y+5)
        n = len(PTC_X[mask])
        if n > 0:
            ws = ptc_weights(x, y, mask)
            vx = np.sum(ws * PTC_WVX[mask]) / np.sum(ws)
            vy = np.sum(ws * PTC_WVY[mask]) / np.sum(ws)
            hmgs = np.linalg.norm(np.cov([PTC_WVX[mask], PTC_WVY[mask]]))
            strs = np.mean(strength(mask))
        else:
            ws = 0
            vx = 0
            vy = 0
            hmgs = 0
            strs = 0

        if np.isnan(hmgs):
            hmgs = 0

        coords_wvx.append(vx)
        coords_wvy.append(vy)

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

    return np.array(coords_x), np.array(coords_y), \
        np.array(coords_wvx), np.array(coords_wvy), np.array(coords_conf)


def plot_ac(ax):
    for i, (x, y, vx, vy) in enumerate(zip(AC_X, AC_Y, AC_VX, AC_VY)):
        ax.scatter(x, y, c='k')
        ax.arrow(x, y, vx, vy, lw=2, head_width=2, head_length=2,
                 ec='k', fc='k')
        cir = plt.Circle((x, y), radius=np.sqrt(AREA*0.8),
                         color='k', fc='none', ls='--', lw=2)
        ax.add_patch(cir)
    ax.set_xlim([0, AREA])
    ax.set_ylim([0, AREA])
    ax.set_aspect('equal')

def plot_particle_samples(ax, sample=10, draw_hdg=None):
    sortidx = np.argsort(PTC_AGE[::sample])[::-1]

    X = PTC_X[::sample][sortidx]
    Y = PTC_Y[::sample][sortidx]
    VX = PTC_WVX[::sample][sortidx]
    VY = PTC_WVY[::sample][sortidx]
    AGE = PTC_AGE[::sample][sortidx]

    if max(AGE) == min(AGE):
        Color = 'gray'
    else:
        Color = cm.Blues(1 + 0.2 - (AGE-min(AGE))/(max(AGE)-min(AGE)))

    ax.scatter(X, Y, s=3, color=Color)

    if draw_hdg:
        for i, (x, y, vx, vy) in enumerate(zip(X, Y, VX, VY)):
            ax.plot([x, x+vx/2], [y, y+vy/2], color='k', alpha=0.5, lw=1)

    ax.set_xlim([0, AREA])
    ax.set_ylim([0, AREA])
    ax.set_aspect('equal')

def plot_wind_grid(ax):
    x, y, vx, vy, conf = wind_grid()

    for x, y, vx, vy in zip(x, y, vx, vy):
        if vx!=0 and vy!=0:
            ax.scatter(x, y, s=15, color='k')
            ax.arrow(x, y, vx, vy, head_width=2, head_length=2,
                     ec='k', fc='k')
        else:
            ax.scatter(x, y, s=15, color='k', facecolors='none')
    ax.set_aspect('equal')

def plot_wind_confidence(ax):
    x, y, vx, vy, conf = wind_grid()

    n = int(np.sqrt(len(x)))
    CS = ax.contourf(
        x.reshape(n, n),
        y.reshape(n, n),
        conf.reshape(n, n),
        np.arange(0.001, 1, 0.1),
        vmin=0, vmax=1,
        cmap=cm.get_cmap(cm.BuGn)
    )
    plt.colorbar(CS, fraction=0.046, pad=0.01)
    ax.set_aspect('equal')


def observe_ac(n):
    # initialize aicraft
    x = np.random.uniform(0, AREA, n)
    y = np.random.uniform(0, AREA, n)

    v = np.random.normal(4, 0.5, n)
    hdg = np.random.normal(0.25*np.pi, 0.05*np.pi, n)
    # hdg = np.random.uniform(0, 2*np.pi, n)

    vx = v * np.sin(hdg)
    vy = v * np.cos(hdg)

    return x, y, vx, vy


fig = plt.figure(figsize=(15, 5))

for step in range(60):
    dt = 1

    # update aircraft
    AC_X, AC_Y, AC_VX, AC_VY = observe_ac(N_AC)

    # update existing particles, random walk motion model
    n = len(PTC_X)
    if n > 0:
        PTC_X = PTC_X + np.random.normal(PTC_WVX, PTC_WALK_SIGMA*AREA, n)
        PTC_Y = PTC_Y + np.random.normal(PTC_WVY, PTC_WALK_SIGMA*AREA, n)
        PTC_AGE = PTC_AGE + 1

    # add new particles
    for x, y, vx, vy in zip(AC_X, AC_Y, AC_VX, AC_VY):
        pxy = np.random.multivariate_normal(
            [x, y],
            [[NEW_PTC_D_SIGMA*AREA, 0], [0, NEW_PTC_D_SIGMA*AREA]],
            N_AC_PTCS
        )

        px = pxy[:, 0]
        py = pxy[:, 1]
        pvx = vx * (1 + np.random.normal(0, PTC_HDG_VARY_SIGMA, N_AC_PTCS))
        pvy = vy * (1 + np.random.normal(0, PTC_HDG_VARY_SIGMA, N_AC_PTCS))

        PTC_X = np.append(PTC_X, px)
        PTC_Y = np.append(PTC_Y, py)
        PTC_WVX = np.append(PTC_WVX, pvx)
        PTC_WVY = np.append(PTC_WVY, pvy)
        PTC_AGE = np.append(PTC_AGE, np.zeros(N_AC_PTCS))
        PTC_X0 = np.append(PTC_X0, x*np.ones(N_AC_PTCS))
        PTC_Y0 = np.append(PTC_Y0, y*np.ones(N_AC_PTCS))

    # resample particle
    idx = resample()
    PTC_X = PTC_X[idx]
    PTC_Y = PTC_Y[idx]
    PTC_WVX = PTC_WVX[idx]
    PTC_WVY = PTC_WVY[idx]
    PTC_AGE = PTC_AGE[idx]
    PTC_X0 = PTC_X0[idx]
    PTC_Y0 = PTC_Y0[idx]

    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    plot_ac(ax1)
    plot_particle_samples(ax2)
    plot_wind_confidence(ax3)
    plot_wind_grid(ax3)
    # plt.tight_layout()
    # plt.savefig('tmp/pwm-sim-%01d.png' % (step+1))
    # plt.close()
    plt.draw()
    plt.waitforbuttonpress(-1)
    plt.clf()
