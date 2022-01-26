import numpy as np
import matplotlib.pyplot as plt

data_1 = np.load('results/13/TD3_RWIP_0.npy')
data_2 = np.load('results/18/TD3_RWIP_0.npy')

i = 14

#plt.plot(data_1[i:])
plt.plot(data_2[i:])
plt.show()