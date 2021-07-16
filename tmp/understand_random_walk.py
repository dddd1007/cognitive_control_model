import numpy as np
import matplotlib.pyplot as plt

n_step = 1000
random_walk = np.zeros((n_step, 2))
origin = np.zeros((1,2))
steps = np.random.choice(a=[-1,1], size = (n_step, 2))
random_walk = np.concatenate([origin, steps]).cumsum(0)

_, ax = plt.subplots(2, 1, figsize=(12,12), constrained_layout=True)
ax[0].plot(random_walk[:,0],c='b',alpha=1,lw=0.25,ls='-')
ax[0].set_title('1D Random Walk', fontsize=20)
ax[1].plot(random_walk[:,0], random_walk[:,1],c='b',alpha=1,lw=0.25,ls='-')
ax[1].set_title('2D Random Walk', fontsize=20);