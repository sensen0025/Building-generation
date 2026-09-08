import numpy as np
z = np.load('/home/linux/redstone_ai/dataset/v2/latent_norm_rot.npz')
print('mean')
print(repr(z['mean'].tolist()))
print('std')
print(repr(z['std'].tolist()))
