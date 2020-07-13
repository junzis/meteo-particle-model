import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

plt.figure(figsize=(10, 10))
m = Basemap(
    lat_0=51.99,
    lon_0=4.38,
    width=600000,
    height=600000,
    resolution="h",
    projection="tmerc",
)
m.fillcontinents(color="#eeeeee")
m.drawcountries(color="#888888", linewidth=2)
m.drawcoastlines(color="#888888", linewidth=2)
plt.tight_layout()
plt.savefig("bg_highres.png", bbox_inches=0)
