# Gas Particle Model for Wind Field Construction
This is the Python library for wind field estimation based on gas particle model. The wind data are computed based on ADS-B and Mode-S using pyModeS library.


To use the model, a minimal example is:

```python
from pwm import ParticleWindModel
import pandas as pd

wind = pd.read_csv('data/wind.csv')
t0 = int(wind['ts'].min())

pwm = ParticleWindModel()
pwm.run(wind, tstart=t0, tend=t0+60)
pwm.plot_all_level()
```

To quickly test the model out of the box, try:

```
$ python run.py
```

Configurable model parameters (with defaults) are:

```python
pwm.lat0 = 51.99
pwm.lon0 = 4.37

pwm.AREA_XY = (-300, 300)           # km
pwm.AREA_Z = (0, 12)                # km

pwm.GRID_BOND_XY = 20               # +/- km
pwm.GRID_BOND_Z = 0.5               # +/- km

pwm.N_AC_PTCS = 500                 # particles per aircraft
pwm.N_MIN_PTC_TO_COMPUTE = 4        # number of particles to compute

pwm.CONF_BOUND = (0.0, 1.0)         # confident normalization

pwm.DECAY_SIGMA = 60.0              # seconds
pwm.NEW_PTC_XY_SIGMA = 4.0          # km
pwm.NEW_PTC_Z_SIGMA = 0.2           # km
pwm.PTC_DIST_STRENGTH_SIGMA = 30.0  # km
pwm.PTC_WALK_XY_SIGMA = 5.0         # km
pwm.PTC_WALK_Z_SIGMA = 0.2          # km
pwm.PTC_VW_VARY_SIGMA = 0.002       # km/s
```
