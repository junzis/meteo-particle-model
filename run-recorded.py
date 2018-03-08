import pandas as pd
import matplotlib.pyplot as plt
import mp_vis
from stream import Stream
import warnings
warnings.filterwarnings("ignore")

stream = Stream(lat0=51.99, lon0=4.37)

stream.mp.N_AC_PTCS = 500
stream.mp.AGING_SIGMA = 180

adsb0 = pd.read_csv('data/adsb_raw_20180101_0900_utc.csv.gz')
ehs0 = pd.read_csv('data/ehs_raw_20180101_0900_utc.csv.gz')

adsb0.loc[:, 'tsr'] = adsb0.ts.round().astype(int)
ehs0.loc[:, 'tsr'] = ehs0.ts.round().astype(int)

ts0 = int(adsb0.tsr.min())
ts1 = int(adsb0.tsr.max())

for t in range(ts0, ts0+100):

    adsb = adsb0[adsb0.tsr == t]
    ehs = ehs0[ehs0.tsr == t]

    stream.process_raw(adsb.ts.tolist(), adsb.msg.tolist(),
                    ehs.ts.tolist(), ehs.msg.tolist(), tnow=t)


    stream.compute_current_weather()
    stream.update_mp_model()

    print("time: %d | n_ptc: %d"  % (t-ts0, len(stream.mp.PTC_X)))

# construct example grid when finish
data = stream.mp.construct()

# display result
plt.figure(figsize=(10, 8))
mp_vis.plot_all_level_wind(stream.mp, data=data, landscape_view=True)
plt.figure(figsize=(10, 8))
mp_vis.plot_all_level_temp(stream.mp, data=data, landscape_view=True)
