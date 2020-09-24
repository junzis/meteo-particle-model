# Meteo-Particle model for wind and temperature field construction using Mode-S data

This is the Python (version 3) library for wind field estimation based on the Meteo-Particle particle model. The wind and temperature is computed from ADS-B and Mode-S data using pyModeS library.


This tool was developed together with the following research paper. If you find it useful for your research, please consider citing:

```
@article{sun2018mp,
  title={Weather field reconstruction using aircraft surveillance data and a novel meteo-particle model},
  author={Sun, Junzi and V{\^u}, Huy and Ellerbroek, Joost and Hoekstra, Jacco M},
  journal={PloS one},
  volume={13},
  number={10},
  pages={e0205029},
  year={2018},
}
```


---

## Dependent libraries

1. You must install [`pyModeS`](https://github.com/junzis/pyModeS) library for ADS-B and Mode-S decoding.
2. You also need the following common scientific libraries: `numpy`, `pandas`, and `matplotlib`.
3. You may install optional `geomag` library, to support the correction of magnetic declination in BDS60 heading.

For a fresh install, run following commands:

```shell
$ pip install git+https://github.com/junzis/pyModeS
$ pip install numpy pandas matplotlib geomag
```

## Code examples

Examples of using the model with recorded data and real-time streaming are given in `run-recoded.py` and `run-realtime.py` file.

To quickly test the model out of the box, try:

```shell
$ python run-recoded.py
```

or if you have access to a ModeSBeast raw stream on TCP port:

```shell
$ python run-realtime.py --server xx.xx.xx.xx --port xxxxx
```


Configurable model parameters (with defaults) are:

```python
AREA_XY = (-300, 300)           # Area - xy, km
AREA_Z = (0, 12)                # Altitude - km

GRID_BOND_XY = 20               # neighborhood xy, +/- km
GRID_BOND_Z = 0.5               # neighborhood z, +/- km
TEMP_Z_BUFFER = 0.2             # neighborhood z (temp),  +/- km

N_AC_PTCS = 300                 # particles per aircraft
N_MIN_PTC_TO_COMPUTE = 10       # number of particles to compute

CONF_BOUND = (0.0, 1.0)         # confident normalization

AGING_SIGMA = 180.0             # Particle aging parameter, seconds
PTC_DIST_STRENGTH_SIGMA = 30.0  # Weighting parameter - distance, km
PTC_WALK_XY_SIGMA = 5.0         # Particle random walk - xy, km
PTC_WALK_Z_SIGMA = 0.1          # Particle random walk - z, km
PTC_VW_VARY_SIGMA = 0.0002      # Particle initialization wind variation, km/s
PTC_TEMP_VARY_SIGMA = 0.1       # Particle initialization temp variation, K

ACCEPT_PROB_FACTOR = 3          # Measurement acceptance probability factor
PTC_WALK_K = 10                 # Particle random walk factor

```

---

## Plots

One minute simulation:
![simulation](data/screenshots/simulation.gif?raw=true)

Wind field from the sample dataset (snapshot at 01/01/2018 09:02 UTC)
![real-wind-field](data/screenshots/recorded_wind_field.png?raw=true)

Temperature field from the sample dataset (snapshot at 01/01/2018 09:02 UTC)
![real-wind-field](data/screenshots/recorded_temp_field.png?raw=true)
