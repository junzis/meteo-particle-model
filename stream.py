
import pandas as pd
import numpy as np
import time
import datetime
import pyModeS as pms
import pprint
from particle_model import ParticleWindModel
from lib import aero

try:
    import geomag
    GEO_MAG_SUPPORT = True
except:
    print('-' * 80)
    print("Warining: Magnatic heading declination libary (geomag) not found, \
        \nConsidering aircraft magnetic heading as true heading. \
        \n(This may lead to errors in wind field.)")
    print('-' * 80)
    GEO_MAG_SUPPORT = False

class Stream():
    def __init__(self, lat0=51.99, lon0=4.37, pwm_ptc=200, pwm_decay=60, pwm_dt=1):

        self.acs = dict()
        self.updated_acs = set()
        self.wind = None

        self.lat0 = lat0
        self.lon0 = lon0

        self.pwm = ParticleWindModel()
        self.pwm.N_AC_PTCS = pwm_ptc
        self.pwm.DECAY_SIGMA = pwm_decay

        self.pwm.AREA_XY = (-300, 300)

        self.t = 0
        self.pwm_t = 0
        self.pwm_dt = pwm_dt


    def process_raw(self, adsb_ts, adsb_msgs, ehs_ts, ehs_msgs, tnow=None):
        """process a chunk of adsb and ehs messages recieved in the same
        time period.
        """
        if tnow is None:
            tnow = time.time()

        self.t = tnow
        self.updated_acs = set()

        # process adsb message
        for t, msg in zip(adsb_ts, adsb_msgs):
            icao = pms.adsb.icao(msg)
            tc = pms.adsb.typecode(msg)

            if icao not in self.acs:
                self.acs[icao] = {}

            self.acs[icao]['t'] = t

            if 1 <= tc <= 4:
                self.acs[icao]['callsign'] = pms.adsb.callsign(msg)

            if (5 <= tc <= 8) or (tc == 19):
                vdata = pms.adsb.velocity(msg)
                if vdata is None:
                    continue

                spd, trk, roc, tag = vdata
                if tag != 'GS':
                    continue

                self.acs[icao]['gs'] = spd
                self.acs[icao]['trk'] = trk
                self.acs[icao]['roc'] = roc
                self.acs[icao]['tv'] = t

            if (5 <= tc <= 18):
                oe = pms.adsb.oe_flag(msg)
                self.acs[icao][oe] = msg
                self.acs[icao]['t'+str(oe)] = t

                if ('tpos' in self.acs[icao]) and (t - self.acs[icao]['tpos'] < 180):
                    # use single message decoding
                    rlat = self.acs[icao]['lat']
                    rlon = self.acs[icao]['lon']
                    latlon = pms.adsb.position_with_ref(msg, rlat, rlon)
                elif ('t0' in self.acs[icao]) and ('t1' in self.acs[icao]) and \
                     (abs(self.acs[icao]['t0'] - self.acs[icao]['t1']) < 10):
                    # use multi message decoding
                    # try:
                    latlon = pms.adsb.position(
                        self.acs[icao][0],
                        self.acs[icao][1],
                        self.acs[icao]['t0'],
                        self.acs[icao]['t1'],
                        self.lat0, self.lon0
                        )
                    # except:
                    #     # mix of surface and airborne position message
                    #     continue
                else:
                    latlon = None

                if latlon is not None:
                    self.acs[icao]['tpos'] = t
                    self.acs[icao]['lat'] = latlon[0]
                    self.acs[icao]['lon'] = latlon[1]
                    self.acs[icao]['alt'] = pms.adsb.altitude(msg)
                    self.updated_acs.update([icao])

        # process ehs message
        for t, msg in zip(ehs_ts, ehs_msgs):
            icao = pms.ehs.icao(msg)

            if icao not in self.acs:
                continue

            bds = pms.ehs.BDS(msg)

            if bds == 'BDS50':
                tas = pms.ehs.tas50(msg)

                if tas:
                    self.acs[icao]['t50'] = t
                    self.acs[icao]['tas'] = tas

            elif bds == 'BDS60':
                ias = pms.ehs.ias60(msg)
                hdg = pms.ehs.hdg60(msg)
                mach = pms.ehs.mach60(msg)

                if ias or hdg or mach:
                    self.acs[icao]['t60'] = t
                    self.acs[icao]['ias'] = ias
                    self.acs[icao]['hdg'] = hdg
                    self.acs[icao]['mach'] = mach

        # clear up old data
        for icao in list(self.acs.keys()):
            if self.t - self.acs[icao]['t'] > 180:
                del self.acs[icao]
                continue

            if ('t50' in self.acs[icao]) and (self.t - self.acs[icao]['t50'] > 5):
                del self.acs[icao]['t50']
                del self.acs[icao]['tas']

            if ('t60' in self.acs[icao]) and (self.t - self.acs[icao]['t60'] > 5):
                del self.acs[icao]['t60']
                del self.acs[icao]['ias']
                del self.acs[icao]['hdg']
                del self.acs[icao]['mach']

        self.compute_current_wind()

    def compute_current_wind(self):
        ts = []
        icaos = []
        lats = []
        lons = []
        alts = []
        vgs = []
        trks = []
        vas = []
        hdgs = []

        for icao in self.updated_acs:   # only last updated
            ac = self.acs[icao]

            if ('tpos' not in ac) or ('tv' not in ac) or ('t60' not in ac) or \
                    ('gs' not in ac) or (ac['hdg'] is None) or (ac['trk'] is None):
                continue

            if self.t - ac['tpos'] < 5 and self.t - ac['t60'] < 5:
                if ('t50' in ac) and (ac['t50'] > ac['t60']):
                    va = ac['tas'] * 0.5144
                else:
                    if (ac['ias']) and (ac['mach'] < 0.3):
                        va = aero.cas2tas(ac['ias'] * 0.5144, ac['alt'] * 0.3048)
                    elif ac['mach']:
                        va = aero.mach2tas(ac['mach'], ac['alt'] * 0.3048)
                    else:
                        continue

                ts.append(ac['tpos'])
                icaos.append(icao)
                lats.append(ac['lat'])
                lons.append(ac['lon'])
                alts.append(ac['alt'])
                vgs.append(ac['gs'] * 0.5144)
                trks.append(np.radians(ac['trk']))
                vas.append(va)
                hdgs.append(np.radians(ac['hdg']))

        if GEO_MAG_SUPPORT:
            d_hdgs = []
            for i, hdg in enumerate(hdgs):
                d_hdg = np.radians(geomag.declination(lats[i], lons[i], alts[i]))
                d_hdgs.append(d_hdg)

            hdgs = hdgs - np.array(d_hdgs)

        vgx = vgs * np.sin(trks)
        vgy = vgs * np.cos(trks)
        vax = vas * np.sin(hdgs)
        vay = vas * np.cos(hdgs)

        vwx = vgx - vax
        vwy = vgy - vay

        self.wind = dict()
        mask = np.isfinite(vwx) & (np.abs(vwx) < 100) & (np.abs(vwy) < 100)
        self.wind['ts'] = np.array(ts)[mask]
        self.wind['icao'] = np.array(icaos)[mask]
        self.wind['lat'] = np.array(lats)[mask]
        self.wind['lon'] = np.array(lons)[mask]
        self.wind['alt'] = np.array(alts)[mask]
        self.wind['vwx'] = vwx[mask]
        self.wind['vwy'] = vwy[mask]


    def update_wind_model(self):
        self.pwm.sample(self.wind, self.pwm_dt)
        self.pwm_t = self.t


    def get_cached_aircraft(self):
        """all aircraft that are stored in memeory (updated within 3 minutes)"""
        return self.acs


    def get_updated_aircraft(self):
        """update aircraft from last iteration"""
        selacs = dict()
        for ac in self.updated_acs:
            selacs[ac] = self.acs[ac]
        return selacs

    def get_current_wind_data(self):
        df = pd.DataFrame.from_dict(self.wind)

        if df.shape[0] == 0:
            return None

        df.loc[:, 'vw'] = np.sqrt(df.vwx**2 + df.vwy**2)
        df.loc[:, 'dw'] = np.degrees(np.arctan2(df.vwx, df.vwy))
        columns = ['ts', 'icao', 'lat', 'lon', 'alt', 'vw', 'dw', 'vwx', 'vwy']
        return df[columns]

    def get_current_wind_model(self):
        return self.pwm, self.pwm_t
