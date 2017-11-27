import os
import sys
import pandas as pd
import numpy as np
import time
import calendar
import pyModeS as pms
from napy import aero

sys.path.insert(0, '../')
import stream

reload(stream)

dataroot = "/mnt/1TB/code_data/wind"

dates = [
    '20170724', '20170725', '20170726', '20170727',
    '20170728', '20170729', '20170730'
]

utc_hours = ['00', '06', '12', '18']        # local timezone CEST: GMT+2h

stm = stream.Stream(pwm_ptc=250, pwm_decay=30)

def extract_adsb_ehs(dates):
    for d in dates:
        print "read adsb %s..." % d
        f_adsb = dataroot + '/adsb/full/ADSB_RAW_' + d + '.csv'
        print "read ehs %s..." % d
        f_ehs = dataroot + '/ehs/full/EHS_RAW_' + d + '.csv'

        df_adsb = pd.read_csv(f_adsb, names=['ts', 'icao', 'tc', 'msg'])
        df_ehs = pd.read_csv(f_ehs, names=['ts', 'icao', 'msg'])

        for h in utc_hours:
            dh = d + h
            ts = calendar.timegm(time.strptime(dh, '%Y%m%d%H'))

            print ts
            t0 = ts - 1800
            t1 = ts + 1800

            df_adsb[df_adsb['ts'].between(t0, t1)] \
                .to_csv(dataroot + '/adsb/ADSB_RAW_%s.csv' % dh, index=False)

            df_ehs[df_ehs['ts'].between(t0, t1)] \
                .to_csv(dataroot + '/ehs/EHS_RAW_%s.csv' % dh, index=False)


def compute_wind(dates):
    for d in dates:
        for h in utc_hours:
            dh = d + h

            if dh in ['2017072400', '2017072406']:
                continue

            print "process %s ..." % dh

            f_adsb = dataroot + '/adsb/ADSB_RAW_' + dh + '.csv'
            f_ehs = dataroot + '/ehs/EHS_RAW_' + dh + '.csv'
            f_wind = dataroot + '/wind_obs/wind_obs_' + dh + '.csv'

            if os.path.isfile(f_wind):
                print "deleting existing file."
                os.remove(f_wind)

            adsb0 = pd.read_csv(f_adsb)
            ehs0 = pd.read_csv(f_ehs)
            adsb0.loc[:, 'tsr'] = adsb0.ts.round().astype(int)
            ehs0.loc[:, 'tsr'] = ehs0.ts.round().astype(int)

            ts0 = adsb0.tsr.min()
            ts1 = adsb0.tsr.max()

            tic = time.time()
            tstart = time.time()

            for t in range(ts0, ts1):
                if t % 60 == 0:
                    tm = time.strftime("%Y-%m-%d %H:%M", time.gmtime(t))
                    toc = time.time()
                    print " %s (%d second)" % (tm, toc-tic)
                    tic = toc

                adsb = adsb0[adsb0.tsr == t]
                ehs = ehs0[ehs0.tsr == t]

                stm.process_raw(adsb.ts.tolist(), adsb.msg.tolist(),
                                ehs.ts.tolist(), ehs.msg.tolist(), tnow=t)

                wind = stm.get_current_wind_data()

                if wind is None:
                    continue

                if os.path.isfile(f_wind):
                    wind.to_csv(f_wind, index=False, header=False, mode='a')
                else:
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

            f_grb = dataroot + '/gfsanl/full/%s_gfs.t%sz.pgrb2.0p25.anl' % (dh, h)

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
            df.to_csv(dataroot + '/gfsanl/gfsanl_%s.csv' % dh, index=False)

# extract_adsb_ehs(dates)
compute_wind(dates)
# extract_gfs_wind(dates)
