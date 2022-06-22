import numpy as np
import matplotlib.pyplot as plt

data_1 = np.load('results/13/TD3_RWIP_0.npy')
data_2 = np.load('results/10/TD3_RWIP_0.npy')

i = 14
#x_axis = np.linspace(i,np.shape(data_1)[0],101-i)
x_axis = np.linspace(i,np.shape(data_1)[0], 20-i)
print(np.shape(data_1)[0])

#plt.plot(data_1[i:])
plt.plot(x_axis,data_1[i:])
plt.show()