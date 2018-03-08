import pandas as pd
import numpy as np
import time
import pyModeS as pms
from mp import MeteoParticleModel
from lib import aero

try:
    import geomag
    GEO_MAG_SUPPORT = True
except:
    print(('-' * 80))
    print("Warining: Magnatic heading declination libary (geomag) not found, \
        \nConsidering aircraft magnetic heading as true heading. \
        \n(This may lead to errors in wind field.)")
    print(('-' * 80))
    GEO_MAG_SUPPORT = False

class Stream():
    def __init__(self, lat0, lon0):

        self.acs = dict()
        self.__ehs_updated_acs = set()
        self.weather = None

        self.lat0 = lat0
        self.lon0 = lon0

        self.mp = MeteoParticleModel(lat0, lon0)

        self.t = 0
        self.mp_t = 0

    def process_raw(self, adsb_ts, adsb_msgs, ehs_ts, ehs_msgs, tnow=None):
        """process a chunk of adsb and ehs messages recieved in the same
        time period.
        """
        if tnow is None:
            tnow = time.time()

        self.t = tnow

        local_ehs_updated_acs_buffer = []

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
                    try:
                        latlon = pms.adsb.position(
                            self.acs[icao][0],
                            self.acs[icao][1],
                            self.acs[icao]['t0'],
                            self.acs[icao]['t1'],
                            self.lat0, self.lon0
                            )
                    except:
                        # mix of surface and airborne position message
                        continue
                else:
                    latlon = None

                if latlon is not None:
                    self.acs[icao]['tpos'] = t
                    self.acs[icao]['lat'] = latlon[0]
                    self.acs[icao]['lon'] = latlon[1]
                    self.acs[icao]['alt'] = pms.adsb.altitude(msg)

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
                    local_ehs_updated_acs_buffer.append(icao)

            elif bds == 'BDS60':
                ias = pms.ehs.ias60(msg)
                hdg = pms.ehs.hdg60(msg)
                mach = pms.ehs.mach60(msg)

                if ias and hdg and mach:
                    self.acs[icao]['t60'] = t
                    self.acs[icao]['ias'] = ias
                    self.acs[icao]['hdg'] = hdg
                    self.acs[icao]['mach'] = mach
                    local_ehs_updated_acs_buffer.append(icao)

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

        self.add_ehs_updated_aircraft(local_ehs_updated_acs_buffer)
        return

    def compute_current_weather(self):
        ts = []
        icaos = []
        lats = []
        lons = []
        alts = []
        temps = []
        vgs = []
        trks = []
        vas = []
        hdgs = []

        update_acs = self.get_updated_aircraft()

        for icao, ac in list(update_acs.items()):   # only last updated
            # ac = self.acs[icao]
            # print(ac['icao'])

            if ('tpos' not in ac) or ('tv' not in ac) or ('t60' not in ac) or \
                    ('t50' not in ac) or ('gs' not in ac):
                continue

            if (self.t - ac['tpos'] > 5) or (self.t - ac['t60'] > 5) or (self.t - ac['t50'] > 5):
                continue

            # print(ac)

            h = ac['alt'] * aero.ft
            vtas  = ac['tas'] * aero.kts
            vias  = ac['ias'] * aero.kts
            mach = ac['mach']

            # p = aero.p0 * np.exp(-1 * aero.Mo * aero.g0 * h / (aero.R * aero.T0) )
            p = 101325 * np.exp(-0.00012 * h)

            if mach < 0.3:
                temp = vtas**2 * p / (vias**2 * aero.rho0 * aero.R)
                rho = p / (aero.R * temp)
                vtas2 = vias * np.sqrt(aero.rho0 / rho)
            else:
                temp = vtas**2 * aero.T0 / (mach**2 * aero.a0**2)
                vtas2 = mach * aero.a0 * np.sqrt(temp / aero.T0)

            va = vtas if ac['t50'] > ac['t60'] else vtas2

            # # compute wind - ISA conditions - old model
            # if ('t50' in ac) and (ac['t50'] > ac['t60']):
            #     va = ac['tas'] * 0.5144
            # else:
            #     if (ac['ias'] is not None) and (ac['mach'] is not None):
            #         if ac['mach'] < 0.3:
            #             va = aero.cas2tas(ac['ias'] * 0.5144, ac['alt'] * 0.3048)
            #         else:
            #             va = aero.mach2tas(ac['mach'], ac['alt'] * 0.3048)
            #     elif (ac['ias'] is not None) and (ac['mach'] is None):
            #         va = aero.cas2tas(ac['ias'] * 0.5144, ac['alt'] * 0.3048)
            #     elif (ac['ias'] is None) and (ac['mach'] is not None):
            #         va = aero.mach2tas(ac['mach'], ac['alt'] * 0.3048)
            #     else:
            #         continue

            ts.append(ac['tpos'])
            icaos.append(icao)
            lats.append(ac['lat'])
            lons.append(ac['lon'])
            alts.append(ac['alt'])
            temps.append(temp)
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

        wx = vgx - vax
        wy = vgy - vay

        self.weather = dict()
        self.weather['ts'] = np.array(ts)
        self.weather['icao'] = np.array(icaos)
        self.weather['lat'] = np.array(lats)
        self.weather['lon'] = np.array(lons)
        self.weather['alt'] = np.array(alts)
        self.weather['wx'] = wx
        self.weather['wy'] = wy
        self.weather['temp'] = temps

        # important to reset the buffer
        self.reset_updated_aircraft()

        return self.weather

    def update_mp_model(self):
        self.mp.sample(self.weather)
        self.mp_t = self.t


    def get_cached_aircraft(self):
        """all aircraft that are stored in memeory (updated within 3 minutes)"""
        return self.acs


    def add_ehs_updated_aircraft(self, acs):
        """add new aircraft to the list"""
        self.__ehs_updated_acs.update(acs)
        return

    def get_updated_aircraft(self):
        """update aircraft from last iteration"""
        updated = dict()
        for ac in self.__ehs_updated_acs:
            if ac not in self.acs:
                continue
            updated[ac] = self.acs[ac]
        return updated

    def reset_updated_aircraft(self):
        """reset the updated icao buffer once been read"""
        self.__ehs_updated_acs = set()


    def get_current_weather_data(self):
        df = pd.DataFrame.from_dict(self.weather)

        if df.shape[0] == 0:
            return None

        df.loc[:, 'vw'] = np.sqrt(df.wx**2 + df.wy**2)
        df.loc[:, 'dw'] = np.degrees(np.arctan2(df.wx, df.wy))
        columns = ['ts', 'icao', 'lat', 'lon', 'alt', 'vw', 'dw', 'wx', 'wy']
        return df[columns]

    def get_current_mp_model(self):
        return self.mp, self.mp_t
