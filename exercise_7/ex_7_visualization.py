from baselines.common import plot_util as pu
folder_name='../baselines/logs/Hopper'
results = pu.load_results(folder_name)

import matplotlib.pyplot as plt
import numpy as np
r = results[0]
#plt.plot(np.cumsum(r.monitor.l), r.monitor.r)
#plt.plot(np.cumsum(r.monitor.l), pu.smooth(r.monitor.r, radius=10))
plt.plot(r.progress['misc/total_timesteps'], r.progress.eprewmean)
plt.show()
