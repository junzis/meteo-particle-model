import pandas as pd
import numpy as np
from lib import aero
import datetime

class MeteoParticleModel():
    def __init__(self, lat0, lon0, tstep=1):
        self.lat0 = lat0
        self.lon0 = lon0

        self.tstep = tstep

        self.AREA_XY = (-300, 300)           # Area - xy, km
        self.AREA_Z = (0, 12)                # Altitude - km

        self.GRID_BOND_XY = 20               # neighborhood xy, +/- km
        self.GRID_BOND_Z = 0.5               # neighborhood z, +/- km
        self.TEMP_Z_BUFFER = 0.2             # neighborhood z (temp),  +/- km

        self.N_AC_PTCS = 300                 # particles per aircraft
        self.N_MIN_PTC_TO_COMPUTE = 10       # number of particles to compute

        self.CONF_BOUND = (0.0, 1.0)         # confident normalization

        self.AGING_SIGMA = 180.0             # Particle aging parameter, seconds
        self.PTC_DIST_STRENGTH_SIGMA = 30.0  # Weighting parameter - distance, km
        self.PTC_WALK_XY_SIGMA = 5.0         # Particle random walk - xy, km
        self.PTC_WALK_Z_SIGMA = 0.1          # Particle random walk - z, km
        self.PTC_VW_VARY_SIGMA = 0.0002      # Particle initialization wind variation, km/s
        self.PTC_TEMP_VARY_SIGMA = 0.1       # Particle initialization temp variation, K

        self.ACCEPT_PROB_FACTOR = 3          # Measurement acceptance probability factor
        self.PTC_WALK_K = 10                 # Particle random walk factor

        self.reset_model()


    def reset_model(self):
        # aicraft
        self.AC_X = np.array([])
        self.AC_Y = np.array([])
        self.AC_Z = np.array([])
        self.AC_WX = np.array([])
        self.AC_WY = np.array([])
        self.AC_TEMP = np.array([])

        # particles
        self.PTC_X = np.array([])            # current position of particles
        self.PTC_Y = np.array([])
        self.PTC_Z = np.array([])
        self.PTC_WX = np.array([])           # particles weather state
        self.PTC_WY = np.array([])
        self.PTC_TEMP = np.array([])
        self.PTC_AGE = np.array([])
        self.PTC_X0 = np.array([])           # origin positions of particles
        self.PTC_Y0 = np.array([])
        self.PTC_Z0 = np.array([])

        # misc.
        self.snapshots = {}


    def resample(self):
        mask1 = self.PTC_X > self.AREA_XY[0] - self.GRID_BOND_XY
        mask1 &= self.PTC_X < self.AREA_XY[1] + self.GRID_BOND_XY
        mask1 &= self.PTC_Y > self.AREA_XY[0] - self.GRID_BOND_XY
        mask1 &= self.PTC_Y < self.AREA_XY[1] + self.GRID_BOND_XY
        mask1 &= self.PTC_Z > self.AREA_Z[0]
        mask1 &= self.PTC_Z < self.AREA_Z[1]

        prob = np.exp(-0.5 * self.PTC_AGE**2 / self.AGING_SIGMA**2)
        choice = np.random.random(len(self.PTC_X))
        mask2 = prob > choice

        mask = mask1 & mask2

        return np.where(mask)[0]


    # def strength(self, mask):
    #     """decaying factor of particles
    #     """
    #     ptc_ages = self.PTC_AGE[mask]
    #     strength = np.exp(-1 * ptc_ages**2 / (2 * self.AGING_SIGMA**2))
    #     return strength

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

        weights = fd * fd0

        return weights

    def scaled_confidence(self, l):
        """kernel function to scale confidence values
        """
        a, b = self.CONF_BOUND
        l = np.array(l)
        lscale = (b - a) * (l - np.min(l)) / (np.nanmax(l) - np.nanmin(l)) + a
        return lscale

    def construct(self, coords=None, xyz=True, confidence=True):
        if coords is not None:
            if xyz:
                coords_xs, coords_ys, coords_zs = coords
            else:
                lat, lon, alt = coords
                bearings = aero.bearing(self.lat0, self.lon0, np.asarray(lat), np.asarray(lon))
                distances = aero.distance(self.lat0, self.lon0, np.asarray(lat), np.asarray(lon))

                coords_xs = distances * np.sin(np.radians(bearings)) / 1000.0
                coords_ys = distances * np.cos(np.radians(bearings)) / 1000.0
                coords_zs = np.asarray(alt) * aero.ft / 1000.0

        else:
            xs = np.arange(self.AREA_XY[0], self.AREA_XY[1]+1, (self.AREA_XY[1]-self.AREA_XY[0])/10)
            ys = np.arange(self.AREA_XY[0], self.AREA_XY[1]+1, (self.AREA_XY[1]-self.AREA_XY[0])/10)
            zs = np.linspace(self.AREA_Z[0]+1, self.AREA_Z[1], 12)

            xx, yy, zz = np.meshgrid(xs, ys, zs)
            coords_xs = xx.flatten()
            coords_ys = yy.flatten()
            coords_zs = zz.flatten()

        coords_wx = []
        coords_wy = []
        coords_temp = []

        coords_ptc_wei = []
        coords_ptc_num = []
        coords_ptc_w_hmg = []
        coords_ptc_t_hmg = []
        coords_ptc_str = []

        for x, y, z in zip(coords_xs, coords_ys, coords_zs):
            mask1 = (self.PTC_X > x - self.GRID_BOND_XY) & (self.PTC_X < x + self.GRID_BOND_XY) \
                    & (self.PTC_Y > y - self.GRID_BOND_XY) & (self.PTC_Y < y + self.GRID_BOND_XY) \
                    & (self.PTC_Z > z - self.GRID_BOND_Z) & (self.PTC_Z < z + self.GRID_BOND_Z) \

            # additional mask for temperature, only originated in similar level
            mask2 = mask1 & (self.PTC_Z0 > z - self.TEMP_Z_BUFFER) & (self.PTC_Z0 < z + self.TEMP_Z_BUFFER)

            n = len(self.PTC_X[mask1])

            if n > self.N_MIN_PTC_TO_COMPUTE:
                w = self.ptc_weights(x, y, z, mask1)
                wsum = np.sum(w)
                if wsum < 1e-100:
                    # incase of all weights becomes almost zero
                    wx = np.nan
                    wy = np.nan
                else:
                    wx = np.sum(w * self.PTC_WX[mask1]) / wsum
                    wy = np.sum(w * self.PTC_WY[mask1]) / wsum

                w2 = self.ptc_weights(x, y, z, mask2)
                wsum2 = np.sum(w2)
                if wsum2 < 1e-100:
                    # incase of all weights becomes almost zero
                    temp = np.nan
                else:
                    temp = np.sum(w2 * self.PTC_TEMP[mask2]) / wsum2

                if confidence:
                    strs = 1 / (np.mean(self.PTC_AGE[mask1]) + 1e-100)
                    w_hmgs = np.linalg.norm(np.cov([self.PTC_WX[mask1], self.PTC_WY[mask1]]))
                    w_hmgs = 0 if np.isnan(w_hmgs) else w_hmgs

                    t_hmgs = np.std(self.PTC_TEMP[mask2])
            else:
                w = 0.0
                wx = np.nan
                wy = np.nan
                temp = np.nan
                if confidence:
                    t_hmgs = 0.0
                    w_hmgs = 0.0
                    strs = 0.0

            coords_wx.append(wx)
            coords_wy.append(wy)
            coords_temp.append(temp)

            if confidence:
                coords_ptc_num.append(n)
                coords_ptc_wei.append(np.mean(w))
                coords_ptc_str.append(strs)
                coords_ptc_t_hmg.append(t_hmgs)
                coords_ptc_w_hmg.append(w_hmgs)

        # compute confidence at each grid point, based on:
        #   particle numbers, mean weights, uniformness of particle headings
        if confidence:
            fw = self.scaled_confidence(coords_ptc_wei)
            fn = self.scaled_confidence(coords_ptc_num)
            fh_w = self.scaled_confidence(coords_ptc_w_hmg)
            fh_t = self.scaled_confidence(coords_ptc_t_hmg)
            fs = self.scaled_confidence(coords_ptc_str)
            coords_w_confs = (fw + fn + fh_w + fs) / 4.0
            coords_t_confs = (fw + fn + fh_t + fs) / 4.0
        else:
            coords_w_confs = None
            coords_t_confs = None

        return np.array(coords_xs), np.array(coords_ys), np.array(coords_zs), \
            np.array(coords_wx), np.array(coords_wy), np.array(coords_temp), \
            np.array(coords_w_confs), np.array(coords_t_confs)

    def prob_ac_accept(self):

        n0 = n1 = len(self.AC_X)

        if len(self.PTC_X) / self.N_AC_PTCS < 10:
            mask = [True] * n0

        else:
            ZLo = self.AC_Z - self.GRID_BOND_Z
            ZHi = self.AC_Z + self.GRID_BOND_Z

            MU_WX = np.array([])
            MU_WY = np.array([])
            MU_TEMP = np.array([])
            STD_WX = np.array([])
            STD_WY = np.array([])
            STD_TEMP = np.array([])

            for zlo, zhi in zip(ZLo, ZHi):
                m = (self.PTC_Z > zlo) & (self.PTC_Z < zhi)
                MU_WX = np.append(MU_WX, np.mean(self.PTC_WX[m]))
                MU_WY = np.append(MU_WY, np.mean(self.PTC_WY[m]))
                STD_WX = np.append(STD_WX, np.std(self.PTC_WX[m]))
                STD_WY = np.append(STD_WY, np.std(self.PTC_WY[m]))

                m2 = (self.PTC_Z0 > zlo) & (self.PTC_Z0 < zhi)
                MU_TEMP = np.append(MU_TEMP, np.mean(self.PTC_TEMP[m2]))
                STD_TEMP = np.append(STD_TEMP, np.std(self.PTC_TEMP[m2]))

            mus = np.array([MU_WX, MU_WY, MU_TEMP]).T
            stds = np.array([STD_WX, STD_WY, STD_TEMP]) * self.ACCEPT_PROB_FACTOR
            cov = np.zeros((3, 3))
            np.fill_diagonal(cov, stds**2)
            x = np.array([self.AC_WX, self.AC_WY, self.AC_TEMP]).T

            try:
                dx = x - mus
                cov_inv = np.linalg.inv(cov)
                prob= np.exp(-0.5 * np.einsum('ij,ij->i', np.dot(dx, cov_inv), dx))
                # prob = np.exp(-0.5 * ((self.AC_WX-MU_WX)**2/((k*STD_WX)**2) + (self.AC_WY-MU_WY)**2/((k*STD_WY)**2)))
                choice = np.random.random(len(prob))
                mask =  prob > choice
                mask[np.isnan(prob)] = True
            except:
                mask = [True] * n0

            # print([int(i) for i in mask])

        self.AC_X = self.AC_X[mask]
        self.AC_Y = self.AC_Y[mask]
        self.AC_Z = self.AC_Z[mask]
        self.AC_WX = self.AC_WX[mask]
        self.AC_WY = self.AC_WY[mask]
        self.AC_TEMP = self.AC_TEMP[mask]
        n1 = len(self.AC_X)

        return n0, n1


    def sample(self, weather):
        weather = pd.DataFrame(weather)
        bearings = aero.bearing(self.lat0, self.lon0, weather['lat'], weather['lon'])
        distances = aero.distance(self.lat0, self.lon0, weather['lat'], weather['lon'])

        weather.loc[:, 'x'] = distances * np.sin(np.radians(bearings)) / 1000.0
        weather.loc[:, 'y'] = distances * np.cos(np.radians(bearings)) / 1000.0
        weather.loc[:, 'z'] = weather['alt'] * aero.ft / 1000.0

        self.AC_X = np.asarray(weather['x'])
        self.AC_Y = np.asarray(weather['y'])
        self.AC_Z = np.asarray(weather['z'])
        self.AC_WX = np.asarray(weather['wx'])
        self.AC_WY = np.asarray(weather['wy'])
        self.AC_TEMP = np.asarray(weather['temp'])

        # add new particles
        self.prob_ac_accept()


        n0 = len(self.PTC_X)
        n_new_ptc = len(self.AC_X) * self.N_AC_PTCS

        self.PTC_X = np.append(self.PTC_X, np.zeros(n_new_ptc))
        self.PTC_Y = np.append(self.PTC_Y, np.zeros(n_new_ptc))
        self.PTC_Z = np.append(self.PTC_Z, np.zeros(n_new_ptc))

        self.PTC_WX = np.append(self.PTC_WX, np.zeros(n_new_ptc))
        self.PTC_WY = np.append(self.PTC_WY, np.zeros(n_new_ptc))
        self.PTC_TEMP = np.append(self.PTC_TEMP, np.zeros(n_new_ptc))
        self.PTC_AGE = np.append(self.PTC_AGE, np.zeros(n_new_ptc))

        self.PTC_X0 = np.append(self.PTC_X0, np.zeros(n_new_ptc))
        self.PTC_Y0 = np.append(self.PTC_Y0, np.zeros(n_new_ptc))
        self.PTC_Z0 = np.append(self.PTC_Z0, np.zeros(n_new_ptc))

        px = np.random.normal(0, self.PTC_WALK_XY_SIGMA/2, n_new_ptc)
        py = np.random.normal(0, self.PTC_WALK_XY_SIGMA/2, n_new_ptc)
        pz = np.random.normal(0, self.PTC_WALK_Z_SIGMA/2, n_new_ptc)

        pwx = np.random.normal(0, self.PTC_VW_VARY_SIGMA, n_new_ptc)
        pwy = np.random.normal(0, self.PTC_VW_VARY_SIGMA, n_new_ptc)
        ptemp = np.random.normal(0, self.PTC_TEMP_VARY_SIGMA, n_new_ptc)

        for i, (x, y, z, wx, wy, temp) in enumerate(zip(self.AC_X, self.AC_Y, self.AC_Z, self.AC_WX, self.AC_WY, self.AC_TEMP)):
            idx0 = i*self.N_AC_PTCS
            idx1 = (i+1) * self.N_AC_PTCS

            self.PTC_X[n0+idx0:n0+idx1] = x + px[idx0:idx1]
            self.PTC_Y[n0+idx0:n0+idx1] = y + py[idx0:idx1]
            self.PTC_Z[n0+idx0:n0+idx1] = z + pz[idx0:idx1]

            self.PTC_WX[n0+idx0:n0+idx1] = wx + pwx[idx0:idx1]
            self.PTC_WY[n0+idx0:n0+idx1] = wy + pwy[idx0:idx1]
            self.PTC_TEMP[n0+idx0:n0+idx1] = temp + ptemp[idx0:idx1]
            self.PTC_AGE[n0+idx0:n0+idx1] = np.zeros(self.N_AC_PTCS)

            self.PTC_X0[n0+idx0:n0+idx1] = x * np.ones(self.N_AC_PTCS)
            self.PTC_Y0[n0+idx0:n0+idx1] = y * np.ones(self.N_AC_PTCS)
            self.PTC_Z0[n0+idx0:n0+idx1] = z * np.ones(self.N_AC_PTCS)

        # update existing particles, random walk motion model
        n1 = len(self.PTC_X)
        if n1 > 0:
            ex = np.random.normal(0, self.PTC_WALK_XY_SIGMA, n1)
            ey = np.random.normal(0, self.PTC_WALK_XY_SIGMA, n1)
            self.PTC_X = self.PTC_X + self.PTC_WALK_K * self.PTC_WX/1000.0 * self.tstep + ex     # 1/1000 m/s -> km/s
            self.PTC_Y = self.PTC_Y + self.PTC_WALK_K * self.PTC_WY/1000.0 * self.tstep + ey
            self.PTC_Z = self.PTC_Z + np.random.normal(0, self.PTC_WALK_Z_SIGMA, n1)
            self.PTC_AGE = self.PTC_AGE + self.tstep

        # cleanup particle
        idx = self.resample()
        self.PTC_X = self.PTC_X[idx]
        self.PTC_Y = self.PTC_Y[idx]
        self.PTC_Z = self.PTC_Z[idx]
        self.PTC_WX = self.PTC_WX[idx]
        self.PTC_WY = self.PTC_WY[idx]
        self.PTC_TEMP = self.PTC_TEMP[idx]
        self.PTC_AGE = self.PTC_AGE[idx]
        self.PTC_X0 = self.PTC_X0[idx]
        self.PTC_Y0 = self.PTC_Y0[idx]
        self.PTC_Z0 = self.PTC_Z0[idx]

        return


    def legacy_run(self, winds, tstart, tend, snapat=None, coords=None, debug=False):
        bearings = aero.bearing(self.lat0, self.lon0, winds['lat'], winds['lon'])
        distances = aero.distance(self.lat0, self.lon0, winds['lat'], winds['lon'])

        winds['x'] = distances * np.sin(np.radians(bearings)) / 1000
        winds['y'] = distances * np.cos(np.radians(bearings)) / 1000
        winds['z'] = winds['alt'] * aero.ft / 1000

        for t in range(tstart, tend, 1):

            if debug:
                if t % 30 == 0:
                    print('time:', t-tstart, '| particles:', len(self.PTC_X))

            if (snapat is not None) and (t > tstart):
                if t in snapat:
                    self.snapshots[t] = self.construct(coords=coords)
                    dt = datetime.datetime.utcfromtimestamp(t).strftime("%Y-%m-%d %H:%M")
                    print("winds grid snapshot at %s (%d)" % (dt, t))

            w = winds[winds.ts.astype(int)==t]

            self.sample(w)


    def save_snapshot(self, t):
        data = self.construct()

        x, y, z = data[0], data[1], data[2]

        distance = np.sqrt(x**2 + y**2) * 1000 + 1e-200
        bearing = np.arcsin(1000 * x / distance)

        lat1, lon1 = aero.position(self.lat0, self.lon0, distance, bearing)
        alt1 = z * 1000 / aero.ft

        df = pd.DataFrame()
        df['lat'] = lat1
        df['lon'] = lon1
        df['alt'] = alt1
        df['windx'] = data[3]
        df['windy'] = data[4]
        df['temp'] = data[5]
        df['wind_confidence'] = data[6]
        df['temp_confidence'] = data[7]
        df.to_csv('data/snapshot_%s.csv' % t, index=False)
