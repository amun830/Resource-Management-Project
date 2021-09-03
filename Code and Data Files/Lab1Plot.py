# Import relevant libraries
import numpy as np
from matplotlib import pyplot as plt

# Read data from files
Year1, Cu = np.genfromtxt('ac_cu.csv', delimiter=',', skip_header = 1).T
Year2, Pressure = np.genfromtxt('ac_p.csv', delimiter=',', skip_header = 1).T
Year3, Extraction = np.genfromtxt('ac_q.csv', delimiter=',', skip_header = 1).T

# Create and configure plot
f,host = plt.subplots(figsize=(16,8))
ax1 = host.twinx()
ax2 = host.twinx()
host.set_xlabel('Time [year]',fontsize=12)
ax2.spines['right'].set_position(('outward', 70))
host.set_title('Pressure, Dissolved Copper and Extraction at Onehunga Aquifer Over Time', fontsize=18)
host.set_ylabel('Dissolved Cu [mg/L]', fontsize=12)
ax1.set_ylabel('Pressure [MPa]',fontsize=12)
ax2.set_ylabel('Extraction Rate [10^6 L/day]',fontsize=12)

# Plot data
p1, = host.plot(Year1, Cu, 'b-', label='Copper')
p2, = ax1.plot(Year2, Pressure, 'k-', label='Pressure')
p3, = ax2.plot(Year3, Extraction, 'g-', label='Extraction')
host.legend(handles=[p1,p2,p3], loc='best')

# Show/save plot
# plt.show()
plt.savefig('Lab1Plot.png')