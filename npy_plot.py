import numpy as np
import matplotlib.pyplot as plt

data_1 = np.load('results/13/TD3_RWIP_0.npy')
data_2 = np.load('results/10/TD3_RWIP_0.npy')

i = 9
x_axis = np.linspace(i,np.shape(data_2)[0],101-i)
print(np.shape(data_2)[0])

#plt.plot(data_1[i:])
plt.plot(x_axis*5000,data_2[i:])
plt.show()