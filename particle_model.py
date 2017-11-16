import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from napy import aero
import time

class ParticleWindModel():
    def __init__(self):
        self.lat0 = 51.99
        self.lon0 = 4.37

        self.AREA_XY = (-300, 300)           # km
        self.AREA_Z = (0, 12)                # km

        self.GRID_BOND_XY = 20               # +/- km
        self.GRID_BOND_Z = 0.5               # +/- km

        self.N_AC_PTCS = 500                 # particles per aircraft
        self.N_MIN_PTC_TO_COMPUTE = 4        # number of particles to compute

        self.CONF_BOUND = (0.0, 1.0)         # confident normlization bound

        self.DECAY_SIGMA = 60.0              # seconds
        self.NEW_PTC_XY_SIGMA = 4.0          # km
        self.NEW_PTC_Z_SIGMA = 0.2           # km
        self.PTC_DIST_STRENGTH_SIGMA = 30.0  # km
        self.PTC_WALK_XY_SIGMA = 5.0         # km
        self.PTC_WALK_Z_SIGMA = 0.2          # km
        self.PTC_VW_VARY_SIGMA = 0.002       # km/s

        self.reset_model()


    def reset_model(self):
        # aicraft
        self.AC_X = np.array([])
        self.AC_Y = np.array([])
        self.AC_Z = np.array([])
        self.AC_WVX = np.array([])
        self.AC_WVY = np.array([])

        # particles
        self.PTC_X = np.array([])            # current position of particles
        self.PTC_Y = np.array([])
        self.PTC_Z = np.array([])
        self.PTC_WVX = np.array([])           # particles' wind state
        self.PTC_WVY = np.array([])
        self.PTC_AGE = np.array([])
        self.PTC_X0 = np.array([])           # origin positions of particles
        self.PTC_Y0 = np.array([])
        self.PTC_Z0 = np.array([])

        # misc.
        self.snapshots = {}


    def resample(self):
        mask = self.PTC_AGE < self.DECAY_SIGMA
        mask &= self.PTC_X > self.AREA_XY[0]
        mask &= self.PTC_X < self.AREA_XY[1]
        mask &= self.PTC_Y > self.AREA_XY[0]
        mask &= self.PTC_Y < self.AREA_XY[1]
        return np.where(mask)[0]

    def strength(self, mask):
        """decaying factor of particles
        """
        ptc_ages = self.PTC_AGE[mask]
        strength = np.exp(-1 * ptc_ages**2 / (2 * self.DECAY_SIGMA**2))
        return strength

    def ptc_weights(self, x0, y0, z0, mask):
        """particle weights are calculated as gaussian function
        of distances of particles to a grid point, particle age,
        and particle distance from its origin.
        """
        ptc_xs = self.PTC_X[mask]
        ptc_ys = self.PTC_Y[mask]
        ptc_zs = self.PTC_Z[mask]

        ptc_x0s = self.PTC_X0[mask]
        ptc_y0s = self.PTC_Y0[mask]
        ptc_z0s = self.PTC_Z0[mask]

        d = np.sqrt((ptc_xs-x0)**2 + (ptc_ys-y0)**2 + (ptc_zs-z0)**2)
        fd = np.exp(-1 * d**2 / (2 * self.PTC_DIST_STRENGTH_SIGMA**2))

        ptc_d0s = np.sqrt((ptc_xs-ptc_x0s)**2 + (ptc_ys-ptc_y0s)**2 + (ptc_zs-ptc_z0s)**2)
        fd0 = np.exp(-1 * ptc_d0s**2 / (2 * self.PTC_DIST_STRENGTH_SIGMA**2))

        fa = self.strength(mask)

        weights = fd * fd0 * fa

        return weights

    def scaled_confidence(self, l):
        """kernel function to scale confidence values
        """
        a, b = self.CONF_BOUND
        l = np.array(l)
        lscale = (b - a) * (l - np.min(l)) / (np.max(l) - np.min(l)) + a
        return lscale

    def wind_grid(self, coords=None, xyz=True, confidence=True):
        if coords is not None:
            if xyz:
                coords_xs, coords_ys, coords_zs = coords
            else:
                lat, lon, alt = coords
                bearings = aero.bearing(self.lat0, self.lon0, lat, lon)
                distances = aero.distance(self.lat0, self.lon0, lat, lon)

                coords_xs = distances * np.sin(np.radians(bearings)) / 1000.0
                coords_ys = distances * np.cos(np.radians(bearings)) / 1000.0
                coords_zs = alt * aero.ft / 1000.0

        else:
            xs = np.arange(self.AREA_XY[0], self.AREA_XY[1]+1, (self.AREA_XY[1]-self.AREA_XY[0])/10)
            ys = np.arange(self.AREA_XY[0], self.AREA_XY[1]+1, (self.AREA_XY[1]-self.AREA_XY[0])/10)
            zs = np.linspace(self.AREA_Z[0]+1, self.AREA_Z[1], 12)

            xx, yy, zz = np.meshgrid(xs, ys, zs)
            coords_xs = xx.flatten()
            coords_ys = yy.flatten()
            coords_zs = zz.flatten()

        coords_wvx = []
        coords_wvy = []

        coords_ptc_wei = []
        coords_ptc_num = []
        coords_ptc_hmg = []
        coords_ptc_str = []

        for x, y, z in zip(coords_xs, coords_ys, coords_zs):
            mask = (self.PTC_X > x - self.GRID_BOND_XY) & (self.PTC_X < x + self.GRID_BOND_XY) \
                    & (self.PTC_Y > y - self.GRID_BOND_XY) & (self.PTC_Y < y + self.GRID_BOND_XY) \
                    & (self.PTC_Z > z - self.GRID_BOND_Z) & (self.PTC_Z < z + self.GRID_BOND_Z) \

            n = len(self.PTC_X[mask])

            if n > self.N_MIN_PTC_TO_COMPUTE:
                ws = self.ptc_weights(x, y, z, mask)
                vx = np.sum(ws * self.PTC_WVX[mask]) / np.sum(ws)
                vy = np.sum(ws * self.PTC_WVY[mask]) / np.sum(ws)

                if confidence:
                    hmgs = np.linalg.norm(np.cov([self.PTC_WVX[mask], self.PTC_WVY[mask]]))
                    hmgs = 0 if np.isnan(hmgs) else hmgs
                    strs = np.mean(self.strength(mask))
            else:
                ws = 0.0
                vx = np.nan
                vy = np.nan
                if confidence:
                    hmgs = 0.0
                    strs = 0.0

            coords_wvx.append(vx)
            coords_wvy.append(vy)

            if confidence:
                coords_ptc_num.append(n)
                coords_ptc_wei.append(np.mean(ws))
                coords_ptc_hmg.append(hmgs)
                coords_ptc_str.append(strs)

        # compute confidence at each grid point, based on:
        #   particle numbers, mean weights, uniformness of particle headings
        if confidence:
            fw = self.scaled_confidence(coords_ptc_wei)
            fn = self.scaled_confidence(coords_ptc_num)
            fh = self.scaled_confidence(coords_ptc_hmg)
            fs = self.scaled_confidence(coords_ptc_str)
            coords_confs = (fw + fn + fh + fs) / 4.0
        else:
            coords_confs = None

        return np.array(coords_xs), np.array(coords_ys), np.array(coords_zs), \
            np.array(coords_wvx), np.array(coords_wvy), np.array(coords_confs)


    def plot_ac(self, ax, zlevel):
        mask = (self.AC_Z > zlevel-self.GRID_BOND_Z) & (self.AC_Z < zlevel+self.GRID_BOND_Z)

        if len(self.AC_X[mask]) < 2:
            return

        for i, (x, y, vx, vy) in enumerate(zip(self.AC_X[mask], self.AC_Y[mask], self.AC_WVX[mask], self.AC_WVY[mask])):

            ax.scatter(x, y, c='k')
            ax.arrow(x, y, vx/3, vy/3, lw=2, head_width=10, head_length=10,
                     ec='k', fc='k')
            # cir = plt.Circle((x, y), radius=np.sqrt(AREA*0.8),
            #                  color='k', fc='none', ls='--', lw=2)
            # ax.add_patch(cir)
        ax.set_xlim(self.AREA_XY)
        ax.set_ylim(self.AREA_XY)
        ax.set_aspect('equal')

    def plot_particle_samples(self, ax, zlevel, sample=10, draw_hdg=None):
        mask = (self.PTC_Z > zlevel-self.GRID_BOND_Z) & (self.PTC_Z < zlevel+self.GRID_BOND_Z)

        if len(self.PTC_X[mask]) < 2:
            return

        sortidx = np.argsort(self.PTC_AGE[mask][::sample])[::-1]

        xs = self.PTC_X[mask][::sample][sortidx]
        ys = self.PTC_Y[mask][::sample][sortidx]
        vxs = self.PTC_WVX[mask][::sample][sortidx]
        vys = self.PTC_WVY[mask][::sample][sortidx]
        ages = self.PTC_AGE[mask][::sample][sortidx]

        if max(ages) == min(ages):
            Color = 'gray'
        else:
            Color = cm.Blues(1 + 0.2 - (ages-min(ages))/(max(ages)-min(ages)))

        ax.scatter(xs, ys, s=3, color=Color)

        if draw_hdg:
            for i, (x, y, vx, vy) in enumerate(zip(xs, ys, vxs, vys)):
                ax.plot([x, x+vx/2], [y, y+vy/2], color='k', alpha=0.5, lw=1)

        ax.set_xlim(self.AREA_XY)
        ax.set_ylim(self.AREA_XY)
        ax.set_aspect('equal')

    def plot_wind_grid_at_z(self, ax, zlevel, data=None):
        x, y, z, vx, vy, conf = self.wind_grid() if data is None else data

        mask = (z==zlevel)

        for x, y, vx, vy in zip(x[mask], y[mask], vx[mask], vy[mask]):
            if np.isfinite(vx) and np.isfinite(vx):
                ax.scatter(x, y, s=4, color='k')
                ax.arrow(x, y, vx/2, vy/2, head_width=10, head_length=10,
                         ec='k', fc='k')
            else:
                ax.scatter(x, y, s=4, color='grey', facecolors='none')
        ax.set_aspect('equal')

    def plot_wind_confidence(self, ax, zlevel, data=None, nxy=None, colorbar=True):
        x, y, z, vx, vy, conf = self.wind_grid() if data is None else data

        mask = (z==zlevel)

        x = x[mask]
        y = y[mask]
        conf = conf[mask]

        if nxy is None:
            nx = ny = int(np.sqrt(len(x)))
        else:
            nx, ny = nxy

        CS = ax.contourf(
            x.reshape(nx, ny),
            y.reshape(nx, ny),
            conf.reshape(nx, ny),
            np.arange(0.001, 1, 0.1),
            # vmin=0, vmax=1,
            cmap=cm.get_cmap(cm.BuGn)
        )

        if colorbar:
            CB = plt.colorbar(CS, fraction=0.046, pad=0.01)
            CB.set_ticks(np.arange(0, 1, 0.1), update_ticks=True)

        ax.set_aspect('equal')

    def plot_plane(self, data=None):
        if data is None:
            data = self.wind_grid()

        zlevels = np.unique(data[2])
        zlevel = zlevels[5]

        fig = plt.figure(figsize=(15, 5))
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)

        self.plot_ac(ax1, zlevel)
        self.plot_particle_samples(ax2, zlevel)
        self.plot_wind_confidence(ax3, zlevel, data)
        self.plot_wind_grid_at_z(ax3, zlevel, data)
        plt.show()

    def plot_all_level(self, data=None, nxy=None, return_plot=False):
        if data is None:
            data = self.wind_grid()

        zlevels = np.unique(data[2])
        n = int(np.sqrt(len(zlevels)))

        # if not return_plot:
        #     plt.figure(figsize=(12, 9))

        for i, z in enumerate(zlevels):
            ax = plt.subplot(n+1, n, i+1)
            self.plot_wind_confidence(ax, z, data=data, nxy=nxy, colorbar=False)
            self.plot_wind_grid_at_z(ax, z, data=data)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title('alt: %d km' % z)

        if not return_plot:
            # plt.tight_layout()
            plt.show()
        else:
            return plt


    def random_ac(self, n):
        # initialize aicraft
        x = np.random.uniform(self.AREA_XY[0], self.AREA_XY[1], n)
        y = np.random.uniform(self.AREA_XY[0], self.AREA_XY[1], n)
        z = np.random.uniform(self.AREA_Z[0], self.AREA_Z[1], n)

        v = np.random.uniform(60, 80, n)
        hdg = np.random.uniform(0.4*np.pi, 0.8*np.pi, n)
        # hdg = np.random.uniform(0, 2*np.pi, n)

        vx = v * np.sin(hdg)
        vy = v * np.cos(hdg)

        return x, y, z, vx, vy


    def dump_paticles_pickle_gz(self):
        import cPickle
        import zlib
        data = {}
        data['param'] = {}

        data['param']['self.lat0'] = self.lat0
        data['param']['self.lon0'] = self.lon0
        data['param']['self.AREA_XY'] = self.AREA_XY
        data['param']['self.AREA_Z'] = self.AREA_Z
        data['param']['self.GRID_BOND_XY'] = self.GRID_BOND_XY
        data['param']['self.GRID_BOND_Z'] = self.GRID_BOND_Z
        data['param']['self.N_AC_PTCS'] = self.N_AC_PTCS
        data['param']['self.N_MIN_PTC_TO_COMPUTE'] = self.N_MIN_PTC_TO_COMPUTE
        data['param']['self.CONF_BOUND'] = self.CONF_BOUND
        data['param']['self.DECAY_SIGMA'] = self.DECAY_SIGMA
        data['param']['self.NEW_PTC_XY_SIGMA'] = self.NEW_PTC_XY_SIGMA
        data['param']['self.NEW_PTC_Z_SIGMA'] = self.NEW_PTC_Z_SIGMA
        data['param']['self.PTC_DIST_STRENGTH_SIGMA'] = self.PTC_DIST_STRENGTH_SIGMA
        data['param']['self.PTC_WALK_XY_SIGMA'] = self.PTC_WALK_XY_SIGMA
        data['param']['self.PTC_WALK_Z_SIGMA'] = self.PTC_WALK_Z_SIGMA
        data['param']['self.PTC_VW_VARY_SIGMA'] = self.PTC_VW_VARY_SIGMA

        data['self.PTC_X'] = self.PTC_X
        data['self.PTC_Y'] = self.PTC_Y
        data['self.PTC_Z'] = self.PTC_Z
        data['self.PTC_WVX'] = self.PTC_WVX
        data['self.PTC_WVY'] = self.PTC_WVY
        data['self.PTC_AGE'] = self.PTC_AGE
        data['self.PTC_X0'] = self.PTC_X0
        data['self.PTC_Y0'] = self.PTC_Y0
        data['self.PTC_Z0'] = self.PTC_Z0

        with open('data/particles.gz', 'wb') as fp:
          fp.write(zlib.compress(cPickle.dumps(data, cPickle.HIGHEST_PROTOCOL), 9))


    def load_particle_pickle_gz(self):
        import cPickle
        import zlib

        with open('data/particles.gz', 'rb') as fp:
            pkl = zlib.decompress(fp.read())
            data = cPickle.loads(pkl)

            return data

    def sample(self, wind, dt=1):
        wind = pd.DataFrame(wind)
        bearings = aero.bearing(self.lat0, self.lon0, wind['lat'], wind['lon'])
        distances = aero.distance(self.lat0, self.lon0, wind['lat'], wind['lon'])

        wind.loc[:, 'x'] = distances * np.sin(np.radians(bearings)) / 1000.0
        wind.loc[:, 'y'] = distances * np.cos(np.radians(bearings)) / 1000.0
        wind.loc[:, 'z'] = wind['alt'] * aero.ft / 1000.0

        self.AC_X = np.asarray(wind['x'])
        self.AC_Y = np.asarray(wind['y'])
        self.AC_Z = np.asarray(wind['z'])
        self.AC_WVX = np.asarray(wind['vwx'])
        self.AC_WVY = np.asarray(wind['vwy'])

        # update existing particles, random walk motion model
        n = len(self.PTC_X)
        if n > 0:
            ex = np.random.normal(0, self.PTC_WALK_XY_SIGMA, n)
            ey = np.random.normal(0, self.PTC_WALK_XY_SIGMA, n)
            self.PTC_X = self.PTC_X + dt*self.PTC_WVX/1000 + ex     # 1/1000 m/s -> km/s
            self.PTC_Y = self.PTC_Y + dt*self.PTC_WVY/1000 + ey
            self.PTC_Z = self.PTC_Z + np.random.normal(0, self.PTC_WALK_Z_SIGMA, n)
            self.PTC_AGE = self.PTC_AGE + dt

        # add new particles
        n_new_ptc = len(self.AC_X) * self.N_AC_PTCS
        self.PTC_X = np.append(self.PTC_X, np.zeros(n_new_ptc))
        self.PTC_Y = np.append(self.PTC_Y, np.zeros(n_new_ptc))
        self.PTC_Z = np.append(self.PTC_Z, np.zeros(n_new_ptc))

        self.PTC_WVX = np.append(self.PTC_WVX, np.zeros(n_new_ptc))
        self.PTC_WVY = np.append(self.PTC_WVY, np.zeros(n_new_ptc))
        self.PTC_AGE = np.append(self.PTC_AGE, np.zeros(n_new_ptc))

        self.PTC_X0 = np.append(self.PTC_X0, np.zeros(n_new_ptc))
        self.PTC_Y0 = np.append(self.PTC_Y0, np.zeros(n_new_ptc))
        self.PTC_Z0 = np.append(self.PTC_Z0, np.zeros(n_new_ptc))

        px = np.random.normal(0, self.NEW_PTC_XY_SIGMA, n_new_ptc)
        py = np.random.normal(0, self.NEW_PTC_XY_SIGMA, n_new_ptc)
        pz = np.random.normal(0, self.NEW_PTC_Z_SIGMA, n_new_ptc)

        pvx = np.random.normal(0, self.PTC_VW_VARY_SIGMA, n_new_ptc)
        pvy = np.random.normal(0, self.PTC_VW_VARY_SIGMA, n_new_ptc)

        for i, (x, y, z, vx, vy) in enumerate(zip(self.AC_X, self.AC_Y, self.AC_Z, self.AC_WVX, self.AC_WVY)):
            idx0 = i*self.N_AC_PTCS
            idx1 = (i+1) * self.N_AC_PTCS

            self.PTC_X[n+idx0:n+idx1] = x + px[idx0:idx1]
            self.PTC_Y[n+idx0:n+idx1] = y + py[idx0:idx1]
            self.PTC_Z[n+idx0:n+idx1] = z + pz[idx0:idx1]

            self.PTC_WVX[n+idx0:n+idx1] = vx * (1 + pvx[idx0:idx1])
            self.PTC_WVY[n+idx0:n+idx1] = vy * (1 + pvx[idx0:idx1])
            self.PTC_AGE[n+idx0:n+idx1] = np.zeros(self.N_AC_PTCS)

            self.PTC_X0[n+idx0:n+idx1] = x * np.ones(self.N_AC_PTCS)
            self.PTC_Y0[n+idx0:n+idx1] = y * np.ones(self.N_AC_PTCS)
            self.PTC_Z0[n+idx0:n+idx1] = z * np.ones(self.N_AC_PTCS)

        # resample particle
        idx = self.resample()
        self.PTC_X = self.PTC_X[idx]
        self.PTC_Y = self.PTC_Y[idx]
        self.PTC_Z = self.PTC_Z[idx]
        self.PTC_WVX = self.PTC_WVX[idx]
        self.PTC_WVY = self.PTC_WVY[idx]
        self.PTC_AGE = self.PTC_AGE[idx]
        self.PTC_X0 = self.PTC_X0[idx]
        self.PTC_Y0 = self.PTC_Y0[idx]
        self.PTC_Z0 = self.PTC_Z0[idx]

        return

    def run(self, winds, tstart, tend, snapat=None, grid=None):
        bearings = aero.bearing(self.lat0, self.lon0, winds['lat'], winds['lon'])
        distances = aero.distance(self.lat0, self.lon0, winds['lat'], winds['lon'])

        winds['x'] = distances * np.sin(np.radians(bearings)) / 1000.0
        winds['y'] = distances * np.cos(np.radians(bearings)) / 1000.0
        winds['z'] = winds['alt'] * aero.ft / 1000.0

        for t in range(tstart, tend, 1):

            # print t, len(self.PTC_X)

            if (snapat is not None) and (grid is not None) and (t > tstart):
                if t in snapat:
                    snapshot = self.wind_grid(grid)
                    self.snapshots[t] = snapshot[3:]  # wx, wy, conf
                    print "winds grid snapshot at:", t

            w = winds[winds.ts.astype(int)==t]

            self.sample(w)

            # time.sleep(0.1)
            # plot_plane()
