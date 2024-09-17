import numpy as np
from cextern.fast_median.fast_median import fast_median_filter_2d
from scipy.ndimage import median_filter
import time

s = int(4000)
img = np.random.random(size=[s,s])
img = img.astype(np.float32)

t = time.time()
img_r = fast_median_filter_2d(img, size=(5,5))
print(f'lib.median_filter_2d_float: {time.time()-t:.2f} s')
t = time.time()
tmp = median_filter(img, size=(5,5))
print(f'ndimage.median_filter: {time.time()-t:.2f} s')
