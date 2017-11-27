import pandas as pd
import numpy as np
import pyModeS as pms
import matplotlib.pyplot as plt
from napy import aero

def aggregate():
    adsb = pd.read_csv('data/adsb.csv')
    ehs = pd.read_csv('data/ehs.csv')

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


    df = adsb.merge(bds60, how='left', on=['icao', 'ts_round']) \
        .merge(bds50, how='left', on=['icao', 'ts_round']) \
        .dropna(subset=['hdg']) \
        [['ts', 'icao', 'lat', 'lon', 'alt', 'gs_adsb', 'trk_adsb',
          'hdg', 'ias', 'mach', 'trk', 'gs', 'tas']]

    df.to_csv('data/aggregated.csv', index=False)


def calc_wind():
    data = pd.read_csv('data/aggregated.csv')

    data.dropna(subset=['hdg', 'trk_adsb'], inplace=True)

    data['vgx'] = data.apply(lambda r: r.gs_adsb * np.sin(np.radians(r.trk_adsb)) * aero.kts, axis=1)
    data['vgy'] = data.apply(lambda r: r.gs_adsb * np.cos(np.radians(r.trk_adsb)) * aero.kts, axis=1)
    data['vax'] = data.apply(lambda r: aero.cas2tas(r.ias*aero.kts, r.alt*aero.ft) * np.sin(np.radians(r.hdg)), axis=1)
    data['vay'] = data.apply(lambda r: aero.cas2tas(r.ias*aero.kts, r.alt*aero.ft) * np.cos(np.radians(r.hdg)), axis=1)

    data['vwx'] = data['vgx'] - data['vax']
    data['vwy'] = data['vgy'] - data['vay']
    data['vw'] = np.sqrt(data['vwx']**2 + data['vwy']**2)
    data['dw'] = np.degrees(np.arctan2(data['vwx'], data['vwy']))

    wind = data[['ts', 'icao', 'lat', 'lon', 'alt', 'vw', 'dw', 'vwx', 'vwy']]
    wind.loc[:, 'vw'] = wind.loc[:, 'vw'].round(2)
    wind.loc[:, 'dw'] = wind.loc[:, 'dw'].round(2)
    wind.loc[:, 'vwx'] = wind.loc[:, 'vwx'].round(2)
    wind.loc[:, 'vwy'] = wind.loc[:, 'vwy'].round(2)
    wind.to_csv('data/wind_1200_1300.csv', index=False)

if __name__ == '__main__':
    # aggregate()

    # calc_wind()

    wind = pd.read_csv('data/wind.csv')

    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.scatter(wind['vw'], wind['alt'], s=0.1)
    plt.xlim([0, 100])

    plt.subplot(132)
    plt.scatter(wind['dw'], wind['alt'], s=0.1)
    plt.xlim([0, 180])

    plt.subplot(133)
    plt.scatter(wind['lat'], wind['lon'], s=0.5)
    plt.tight_layout()
    plt.show()
