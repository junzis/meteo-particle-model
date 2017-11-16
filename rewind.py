import os
import pandas as pd
import numpy as np
import time
import pyModeS as pms
from napy import aero
import stream
import argparse
import sys

reload(stream)

parser = argparse.ArgumentParser()
parser.add_argument('--dumpdir', help='raw messages directory', required=True)
parser.add_argument('--year', required=True)
parser.add_argument('--month', required=True)
parser.add_argument('--day', required=True)
parser.add_argument('--trange', help='time stamp range', nargs='+', type=int, required=False)

args = parser.parse_args()

dumpdir = args.dumpdir
year = int(args.year)
month = int(args.month)
day = int(args.day)
ts0, ts1 = args.trange if args.trange else (None, None)

stm = stream.Stream(pwm_ptc=250, pwm_decay=30)

data_adsb = dumpdir + "/%04d/%04d_%02d/ADSB_RAW_%04d%02d%02d.csv.gz" % \
    (year, year, month, year, month, day)

data_ehs = dumpdir + "/%04d/%04d_%02d/EHS_RAW_%04d%02d%02d.csv.gz" % \
    (year, year, month, year, month, day)

if ts0 and ts1:
    fout = dumpdir + '/decoded_with_wind_%d%02d%02d_%d-%d.csv' % (year, month, day, ts0, ts1)
else:
    fout = dumpdir + '/decoded_with_wind_%d%02d%02d.csv' % (year, month, day)

# # --- test here ---
# data_adsb = '/mnt/8TB/test_adsb_raw.csv'
# data_ehs = '/mnt/8TB/test_ehs_raw.csv'

print 'loading adsb', data_adsb
adsb0 = pd.read_csv(data_adsb, names=['ts', 'icao', 'tc', 'msg'])
print "adsb loaded"

print 'loading ehs', data_ehs
ehs0 = pd.read_csv(data_ehs, names=['ts', 'icao', 'msg'])
print "ehs loaded"

if os.path.isfile(fout):
    print "deleting existing file."
    os.remove(fout)

if ts0 and ts1:
    adsb0 = adsb0[adsb0.ts.between(ts0, ts1)]
    ehs0 = ehs0[ehs0.ts.between(ts0, ts1)]

adsb0['tsr'] = adsb0.ts.round().astype(int)
ehs0['tsr'] = ehs0.ts.round().astype(int)

ts0 = adsb0.tsr.min()
ts1 = adsb0.tsr.max()

tic = time.time()
tstart = time.time()

for t in range(ts0, ts1):
    if t % 60 == 0:
        tm = time.strftime("%Y-%m-%d %H:%M", time.gmtime(t))
        toc = time.time()
        print "%s (%d second)" % (tm, toc-tic)
        tic = toc

    adsb = adsb0[adsb0.tsr == t]
    ehs = ehs0[ehs0.tsr == t]

    stm.process_raw(adsb.ts.tolist(), adsb.msg.tolist(),
                    ehs.ts.tolist(), ehs.msg.tolist(), tnow=t)

    stm.update_wind_field()

    pwm, tpwm = stm.get_current_wind_model()
    # print "time:", t, "| icaos:", len(stm.get_updated_aircraft().keys()), "| n_ptc:", len(stm.pwm.PTC_X)

    acs = pd.DataFrame.from_dict(stm.get_updated_aircraft(), orient='index')
    acs['icao'] = acs.index
    acs['vwx'] = np.nan
    acs['vwy'] = np.nan

    try:
        wdata = pwm.wind_grid(coords=[acs.lat, acs.lon, acs.alt], xyz=False, confidence=False)
        vwx = np.round(wdata[3], 2)
        vwy = np.round(wdata[4], 2)

        acs.loc[:, ['vwx', 'vwy']] = np.array([vwx, vwy]).T

        data = acs[['t', 'icao', 'tpos', 'lat', 'lon', 'alt', 'tv', 'gs', 'trk',
                    't50', 'tas', 't60', 'ias', 'mach', 'hdg', 'vwx', 'vwy']]

        data.dropna(subset=['tpos', 'tv'], inplace=True)
    except Exception, err:
        print err
        continue

    if os.path.isfile(fout):
        data.to_csv(fout, index=False, header=False, mode='a')
    else:
        data.to_csv(fout, index=False)


tend = time.time()
ttotal = (tend-tstart) / 3600.0
print "Total time: %.2fh" % ttotal
