import pandas as pd
import numpy as np
import time
import calendar
import subprocess
import pyModeS as pms
from napy import aero

import sys
sys.path.insert(0, '../')
from config import dataroot

dates = [
    '20170724', '20170725', '20170726', '20170727',
    '20170728', '20170729', '20170730'
]

utc_hours = ['00', '06', '12', '18']        # local timezone CEST: GMT+2h

def extract_adsb_ehs(dates):
    for d in dates:
        print "read adsb %s..." % d
        f_adsb = dataroot + 'wind_paper/adsb/full/ADSB_RAW_' + d + '.csv'
        print "read ehs %s..." % d
        f_ehs = dataroot + 'wind_paper/ehs/full/EHS_RAW_' + d + '.csv'

        df_adsb = pd.read_csv(f_adsb, names=['ts', 'icao', 'tc', 'msg'])
        df_ehs = pd.read_csv(f_ehs, names=['ts', 'icao', 'msg'])

        for h in utc_hours:
            dh = d + h
            ts = calendar.timegm(time.strptime(dh, '%Y%m%d%H'))

            print ts
            t0 = ts - 1800
            t1 = ts + 1800

            df_adsb[df_adsb['ts'].between(t0, t1)] \
                .to_csv(dataroot + 'wind_paper/adsb/ADSB_RAW_%s.csv' % dh, index=False)

            df_ehs[df_ehs['ts'].between(t0, t1)] \
                .to_csv(dataroot + 'wind_paper/ehs/EHS_RAW_%s.csv' % dh, index=False)


def decode_adsb(dates):
    for d in dates:
        for h in utc_hours:
            dh = d + h

            print "decoding adsb %s ..." % dh

            f_in = dataroot + 'wind_paper/adsb/ADSB_RAW_' + dh + '.csv'
            f_out = dataroot + 'wind_paper/adsb/ADSB_DECODED_' + dh + '.csv'

            subprocess.call('python decode_adsb_multi_process.py %s %s' % (f_in, f_out),
                            shell=True)


def compute_wind(dates):
    for d in dates:
        for h in utc_hours:
            dh = d + h

            print "process %s ..." % dh

            f_adsb = dataroot + 'wind_paper/adsb/ADSB_DECODED_' + dh + '.csv'
            f_ehs = dataroot + 'wind_paper/ehs/EHS_RAW_' + dh + '.csv'
            f_wind = dataroot + 'wind_paper/wind_obs/wind_obs_' + dh + '.csv'

            adsb = pd.read_csv(f_adsb)
            ehs = pd.read_csv(f_ehs)

            icaos = adsb.icao.unique()
            ehs['bds'] = ehs.msg.apply(pms.ehs.BDS)

            ehs1 = ehs[ehs['icao'].isin(icaos)].copy()

            bds60 = ehs1[ehs1['bds']=='BDS60'].copy()
            bds50 = ehs1[ehs1['bds']=='BDS50'].copy()

            bds60['hdg'] = bds60.msg.apply(pms.ehs.hdg60)
            bds60['ias'] = bds60.msg.apply(pms.ehs.ias60)
            bds60['mach'] = bds60.msg.apply(pms.ehs.mach60)
            bds60['ts_round'] = bds60['ts'].round().astype(int)
            bds60.drop(['ts'], axis=1, inplace=True)
            bds60.drop_duplicates(subset=['ts_round', 'icao'], inplace=True)

            bds50['trk'] = bds50.msg.apply(pms.ehs.trk50)
            bds50['gs'] = bds50.msg.apply(pms.ehs.gs50)
            bds50['tas'] = bds50.msg.apply(pms.ehs.tas50)
            bds50['ts_round'] = bds50['ts'].round().astype(int)
            bds50.drop(['ts'], axis=1, inplace=True)
            bds50.drop_duplicates(subset=['ts_round', 'icao'], inplace=True)

            adsb['trk_adsb'] = adsb['hdg']
            adsb['gs_adsb'] = adsb['spd']
            adsb.drop(['hdg', 'spd'], axis=1, inplace=True)
            adsb['ts_round'] = adsb['ts'].round().astype(int)

            merged = adsb.merge(bds60, how='left', on=['icao', 'ts_round']) \
                .merge(bds50, how='left', on=['icao', 'ts_round']) \
                .dropna(subset=['hdg']) \
                [['ts', 'icao', 'lat', 'lon', 'alt', 'gs_adsb', 'trk_adsb',
                  'hdg', 'ias', 'mach', 'trk', 'gs', 'tas']]

            merged.dropna(subset=['hdg', 'trk_adsb'], inplace=True)

            merged['vgx'] = merged.apply(lambda r: r.gs_adsb * np.sin(np.radians(r.trk_adsb)) * aero.kts, axis=1)
            merged['vgy'] = merged.apply(lambda r: r.gs_adsb * np.cos(np.radians(r.trk_adsb)) * aero.kts, axis=1)
            merged['vax'] = merged.apply(lambda r: aero.cas2tas(r.ias*aero.kts, r.alt*aero.ft) * np.sin(np.radians(r.hdg)), axis=1)
            merged['vay'] = merged.apply(lambda r: aero.cas2tas(r.ias*aero.kts, r.alt*aero.ft) * np.cos(np.radians(r.hdg)), axis=1)

            merged['vwx'] = merged['vgx'] - merged['vax']
            merged['vwy'] = merged['vgy'] - merged['vay']
            merged['vw'] = np.sqrt(merged['vwx']**2 + merged['vwy']**2)
            merged['dw'] = np.degrees(np.arctan2(merged['vwx'], merged['vwy']))
            merged['ts_utc'] = merged['ts'] - 7200

            wind = merged[['ts', 'ts_utc', 'icao', 'lat', 'lon', 'alt', 'vw', 'dw', 'vwx', 'vwy']]
            wind.loc[:, 'vw'] = wind.loc[:, 'vw'].round(2)
            wind.loc[:, 'dw'] = wind.loc[:, 'dw'].round(2)
            wind.loc[:, 'vwx'] = wind.loc[:, 'vwx'].round(2)
            wind.loc[:, 'vwy'] = wind.loc[:, 'vwy'].round(2)
            wind.to_csv(f_wind, index=False)

def extract_gfs_wind(dates):
    import pygrib
    import numpy as np

    lat_range = [50, 54]
    lon_range = [0, 8]

    lats = np.arange(lat_range[0], lat_range[1]+0.01, 0.25)
    lons = np.arange(lon_range[0], lon_range[1]+0.01, 0.25)

    for d in dates:
        for h in utc_hours:
            dh = d + h

            print "process %s ..." % dh

            f_grb = dataroot + 'wind_paper/gfsanl/full/%s_gfs.t%sz.pgrb2.0p25.anl' % (dh, h)

            grb = pygrib.open(f_grb)
            grb_wind_v = grb.select(shortName="v", typeOfLevel=['isobaricInhPa'])
            grb_wind_u = grb.select(shortName="u", typeOfLevel=['isobaricInhPa'])

            latlons = grb_wind_u[0].latlons()
            lats = latlons[0]
            lons = (latlons[1] + 180) % 360.0 - 180.0

            latmask = (latlons[0][:, 0] > lat_range[0]) & (latlons[0][:, 0] < lat_range[1])
            lonmask = (latlons[1][0, :] > lon_range[0]) & (latlons[1][0, :] < lon_range[1])

            latidx = np.where(latmask)[0]
            lonidx = np.where(lonmask)[0]

            idx = np.meshgrid(latidx, lonidx)

            data = {'lat':[], 'lon': [], 'alt':[], 'wvx':[], 'wvy':[]}

            for grbu, grbv in zip(grb_wind_u, grb_wind_v):
                level = grbu.level

                p = level * 100
                h = (1 - (p / 101325.0)**0.190264) * 44330.76923
                alt = int(h / aero.ft)

                if alt < 44000:
                    lat1d = latlons[0][idx].flatten()
                    lon1d = latlons[1][idx].flatten()
                    wvx1d = grbu.values[idx].flatten()
                    wvy1d = grbv.values[idx].flatten()
                    data['lat'].extend(lat1d)
                    data['lon'].extend(lon1d)
                    data['wvx'].extend(wvx1d)
                    data['wvy'].extend(wvy1d)
                    data['alt'].extend( alt * np.ones(len(lat1d)) )

            df = pd.DataFrame(data)
            df.to_csv(dataroot + 'wind_paper/gfsanl/gfsanl_%s.csv' % dh, index=False)

# extract_adsb_ehs(dates)
# decode_adsb(dates)
# compute_wind(dates)
# extract_gfs_wind(dates)
