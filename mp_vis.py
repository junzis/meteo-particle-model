import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm

root = os.path.dirname(os.path.realpath(__file__))
bg = plt.imread(root+"/extra/bg.png")

def plot_ac(mp, ax, zlevel):
    mask = (mp.AC_Z > zlevel-mp.GRID_BOND_Z) & (mp.AC_Z < zlevel+mp.GRID_BOND_Z)

    if len(mp.AC_X[mask]) < 2:
        return

    for i, (x, y, vx, vy) in enumerate(zip(mp.AC_X[mask], mp.AC_Y[mask], mp.AC_WX[mask], mp.AC_WY[mask])):

        ax.scatter(x, y, c='k')
        ax.arrow(x, y, vx/3, vy/3, lw=2, head_width=10, head_length=10,
                 ec='k', fc='k')
        # cir = plt.Circle((x, y), radius=np.sqrt(AREA*0.8),
        #                  color='k', fc='none', ls='--', lw=2)
        # ax.add_patch(cir)
    ax.set_xlim(mp.AREA_XY)
    ax.set_ylim(mp.AREA_XY)
    ax.set_aspect('equal')

def plot_particle_samples(mp, ax, zlevel, sample=10, draw_hdg=None):
    mask = (mp.PTC_Z > zlevel-mp.GRID_BOND_Z) & (mp.PTC_Z < zlevel+mp.GRID_BOND_Z)

    if len(mp.PTC_X[mask]) < 2:
        return

    sortidx = np.argsort(mp.PTC_AGE[mask][::sample])[::-1]

    xs = mp.PTC_X[mask][::sample][sortidx]
    ys = mp.PTC_Y[mask][::sample][sortidx]
    vxs = mp.PTC_WX[mask][::sample][sortidx]
    vys = mp.PTC_WY[mask][::sample][sortidx]
    ages = mp.PTC_AGE[mask][::sample][sortidx]

    if max(ages) == min(ages):
        Color = 'gray'
    else:
        Color = cm.Blues(1 + 0.2 - (ages-min(ages))/(max(ages)-min(ages)))

    ax.scatter(xs, ys, s=3, color=Color)

    if draw_hdg:
        for i, (x, y, vx, vy) in enumerate(zip(xs, ys, vxs, vys)):
            ax.plot([x, x+vx/2], [y, y+vy/2], color='k', alpha=0.5, lw=1)

    ax.set_xlim(mp.AREA_XY)
    ax.set_ylim(mp.AREA_XY)
    ax.set_aspect('equal')


def plot_wind_grid_at_z(mp, ax, zlevel, data=None, barbs=False):
    xs, ys, zs, vxs, vys, temps, confws, confts = mp.construct() if data is None else data

    mask1 = (zs==zlevel) & np.isfinite(vxs)
    mask2 = (zs==zlevel) & np.isnan(vxs)

    ax.imshow(bg, extent=[-300, 300, -300, 300])

    ax.scatter(xs[mask1], ys[mask1], s=4, color='k')
    ax.scatter(xs[mask2], ys[mask2], s=4, color='grey', facecolors='none')
    
    if barbs:
        ax.barbs(xs[mask1], ys[mask1], vxs[mask1], vys[mask1], length=6, barb_increments={'half': 2.572, 'full': 5.144, 'flag': 25.722})
    else:
        ax.quiver(xs[mask1], ys[mask1], vxs[mask1]*0.7, vys[mask1]*0.7,
                  color='k')
    
    vmean = np.mean(np.sqrt(vxs[mask1]**2 + vys[mask1]**2))
    vmean = 0 if np.isnan(vmean) else vmean

    ax.set_aspect('equal')
    ax.set_title('H: %d km | $\\bar v_w$: %d m/s' % (zlevel, vmean),
                 fontsize=10)


def plot_wind_confidence(mp, ax, zlevel, data=None, nxy=None, colorbar=True):
    xs, ys, zs, vxs, vys, temps, confws, confts = mp.construct() if data is None else data

    mask = (zs==zlevel)

    x = xs[mask]
    y = ys[mask]
    confw = confws[mask]

    if nxy is None:
        nx = ny = int(np.sqrt(len(x)))
    else:
        nx, ny = nxy

    CSw = ax.contourf(
        x.reshape(nx, ny),
        y.reshape(nx, ny),
        confw.reshape(nx, ny),
        levels=np.linspace(0, 1, 10),
        # vmin=0, vmax=1,
        cmap=cm.get_cmap(cm.BuGn),
        alpha=0.6
    )

    if colorbar:
        CB = plt.colorbar(CSw, fraction=0.046, pad=0.01)
        CB.set_ticks(np.linspace(0, 1, 10), update_ticks=True)

    ax.set_aspect('equal')

def plot_temperature_at_z(mp, ax, zlevel, data=None, nxy=None, colorbar=True):
    xs, ys, zs, vxs, vys, temps, confws, confts = mp.construct() if data is None else data

    temps = temps - 273.15
    tmin = np.nanmean(temps)
    tmax = np.nanmax(temps)

    mask = (zs==zlevel)

    x = xs[mask]
    y = ys[mask]
    temp = temps[mask]
    conft = confts[mask]

    if nxy is None:
        nx = ny = int(np.sqrt(len(x)))
    else:
        nx, ny = nxy

    norm = matplotlib.colors.Normalize(vmin=tmin, vmax=tmax)
    color_map = cm.get_cmap('jet')
    color = color_map(norm(temp))
    color[np.isnan(temp)] = (1,1,1,1)

    tmean = np.nanmean(temp)
    tmean = 'n/a' if np.isnan(tmean) else "%d C$^\circ$" % tmean

    ax.imshow(bg, extent=[-300, 300, -300, 300])

    ax.scatter(x, y, color=color)
    # ax.contourf(
    #     x.reshape(nx, ny),
    #     y.reshape(nx, ny),
    #     temp.reshape(nx, ny),
    #     cmap=cm.get_cmap(cm.Reds),
    #     alpha=0.6
    # )
    ax.set_aspect('equal')
    ax.set_title('H: %d km | $\\bar T$: %s' % (zlevel, tmean),
                 fontsize=10)


def plot_all_level_wind(mp, data=None, nxy=None, return_plot=False, landscape_view=False):
    if data is None:
        data = mp.construct()

    zlevels = np.unique(data[2])
    n = int(np.sqrt(len(zlevels)))

    for i, z in enumerate(zlevels):
        if landscape_view:
            ax = plt.subplot(n, n+1, i+1)
        else:
            ax = plt.subplot(n+1, n, i+1)
        plot_wind_confidence(mp, ax, z, data=data, nxy=nxy, colorbar=False)
        plot_wind_grid_at_z(mp, ax, z, data=data)
        ax.set_xticks([])
        ax.set_yticks([])

    if not return_plot:
        plt.tight_layout()
        plt.show()
    else:
        return plt

def plot_all_level_temp(mp, data=None, nxy=None, return_plot=False, landscape_view=False):
    if data is None:
        data = mp.construct()

    zlevels = np.unique(data[2])
    n = int(np.sqrt(len(zlevels)))

    for i, z in enumerate(zlevels):
        if landscape_view:
            ax = plt.subplot(n, n+1, i+1)
        else:
            ax = plt.subplot(n+1, n, i+1)
        plot_temperature_at_z(mp, ax, z, data=data, nxy=nxy, colorbar=False)
        ax.set_xticks([])
        ax.set_yticks([])

    if not return_plot:
        plt.tight_layout()
        plt.show()
    else:
        return plt

def draw_map():
    from mpl_toolkits.basemap import Basemap
    lat0 = 51.989884
    lon0 = 4.375374
    m = Basemap(width=600000, height=600000, resolution='i',
                projection='stere', lat_0=lat0, lon_0=lon0)
    m.drawcoastlines(color='grey', linewidth=1)
    m.drawcountries(color='grey', linewidth=1)
    m.fillcontinents(color='#eeeeee')
    plt.show()


def gpr(res):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel, Matern

    xs, ys, zs, data = res

    nlows = [5, 5, 1]

    for d in data:

        if d.shape[0] < 20:
            continue

        X = d[:, 0:1]
        X_ = np.arange(min(X), 300, 0.1).reshape(-1, 1)
        Y = d[:, 1:]

        for i in range(3):
            y = Y[:, i]

            k = ConstantKernel() \
                + 10**2 * RBF(length_scale=10.0, length_scale_bounds=(10.0, 100.0)) \
                + WhiteKernel(noise_level_bounds=(nlows[i], 20.0))


            gpr = GaussianProcessRegressor(kernel=k)
            gpr.fit(X, y)
            print(gpr.kernel_)

            y_pred, y_std = gpr.predict(X_, return_std=True)

            plt.subplot(3, 1, i+1)
            plt.scatter(X, y, c='k', s=5)
            plt.plot(X_, y_pred)
            plt.fill_between(X_[:, 0], y_pred - y_std, y_pred + y_std,
                             alpha=0.5, color='k')
            plt.xlim(X_.min(), X_.max())
            plt.tight_layout()
        plt.draw()
        plt.waitforbuttonpress(-1)
        plt.clf()
