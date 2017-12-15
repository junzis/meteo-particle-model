import pandas as pd
import matplotlib.pyplot as plt
from particle_model import ParticleWindModel

wind = pd.read_csv('data/wind.csv')
t0 = int(wind['ts'].min())

# initialize the particle model
pwm = ParticleWindModel()

# run model
pwm.run(wind, tstart=t0, tend=t0+120, debug=True)

# display result
plt.figure(figsize=(12, 9))
pwm.plot_all_level(landscape_view=True)
