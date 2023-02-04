import sys
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.collections import PolyCollection

df_cpu = pd.read_csv(sys.argv[1]).set_index('datetime')
df_devices = pd.read_csv(sys.argv[2]).set_index('datetime')

cpu = df_cpu['%user']
device = df_devices['%util']
print(pd.concat([cpu, device], axis=1))
df = pd.concat([cpu, device], axis=1)
df.plot(title="CPU/IO intensive")
plt.savefig("intensive.png")
