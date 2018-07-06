"""
This is a example script to use the MP model on recorded Mode-S data
"""

import pandas as pd
import matplotlib.pyplot as plt
import mp_vis
from stream import Stream
import warnings
warnings.filterwarnings("ignore")

# specify the receiver location, the example receiver is located at TU Delft
stream = Stream(lat0=51.99, lon0=4.37, correction=True)

# you can set up the MP model parameters
stream.mp.N_AC_PTCS = 250
stream.mp.AGING_SIGMA = 180

# read the message dumps
adsb0 = pd.read_csv('data/adsb_raw_20180101_0900_utc.csv.gz')
ehs0 = pd.read_csv('data/ehs_raw_20180101_0900_utc.csv.gz')

# rounding up the timestamp to 1 second for batch process
adsb0.loc[:, 'tsr'] = adsb0.ts.round().astype(int)
ehs0.loc[:, 'tsr'] = ehs0.ts.round().astype(int)

ts0 = int(adsb0.tsr.min())
ts1 = int(adsb0.tsr.max())

for t in range(ts0, ts0+100):

    adsb = adsb0[adsb0.tsr == t]
    ehs = ehs0[ehs0.tsr == t]

    stream.process_raw(adsb.ts.tolist(), adsb.msg.tolist(),
                    ehs.ts.tolist(), ehs.msg.tolist(), tnow=t)

    # compute_current_weather() and update_mp_model() must be run in following sequence
    wd = stream.compute_current_weather()   # weather data also returned
    stream.update_mp_model()

    print("time: %d | n_ptc: %d"  % (t-ts0, len(stream.mp.PTC_X)))


# take an example snapshot
stream.mp.save_snapshot(t)

# construct example grid when finish
data = stream.mp.construct()

# display result
plt.figure(figsize=(10, 8))
mp_vis.plot_all_level_wind(stream.mp, data=data, landscape_view=True)
plt.figure(figsize=(10, 8))
mp_vis.plot_all_level_temp(stream.mp, data=data, landscape_view=True)
