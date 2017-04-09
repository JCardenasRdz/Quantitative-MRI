#standard module
import matplotlib.pyplot as plt
import numpy as np

# Package created by julio
from Relaxation import T1 # import module T2
import importlib; importlib.reload(T1) # reload in case we change it


# plot clean data
TR = np.linspace(0,10,100)
signal = T1.T1rec(TR,T1=3)
plt.plot(TR, signal)
signal = T1.add_noise(signal,.1)
plt.plot(TR, signal,'o')

R = T1.fit(TR, signal)

plt.plot(TR, T1.T1rec(TR,T1=R.x[0],Mz=R.x[1]),'--')
plt.legend( ('Zero Noise','Observed','Predicted') )




