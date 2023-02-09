import sys
import pandas as pd
import matplotlib.pyplot as plt


df_cpu = pd.read_csv(sys.argv[1]).set_index('datetime')
df_devices = pd.read_csv(sys.argv[2]).set_index('datetime')

cpu = df_cpu['%user']
device = df_devices['%util']

print(pd.concat([cpu, device], axis=1))
df = pd.concat([cpu, device], axis=1)
df.plot(title="CPU/IO intensive")

plt.savefig(F"{sys.argv[3]}/intensive.png")
