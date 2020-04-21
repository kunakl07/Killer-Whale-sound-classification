from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from skimage.draw import line, polygon
import time
from skimage.feature import hog
import concurrent.futures
import sys


chunk = int(sys.argv[1])


basePath = '../data/'
labels = np.load(basePath + 'whale_trainlabels.npy')

# Asume 160 x 128 image

length = np.arange(5, 101, 5)  # 11ms advance -> 55 to 1100ms lengths
height = np.arange(1, 76, 5)  # 3.9Hz per coefficient 0 to 1111ms lengths

print('len length', len(length))
print('len height', len(height))
print('total N templates' ,len(length)*len(height))

templates = []

#size added to edges of template
addsize = 6

for h in height:
    for l in length:

        tm = -1 * np.ones((l+addsize, h+addsize))
        startpoint = np.array([int(addsize/2), int(addsize/2)])
        endpoint = startpoint + np.array([l, h])

        rr, cc = line(startpoint[0], startpoint[1], endpoint[0]-1, endpoint[1]-1)
        tm[rr, cc] = 1
        templates.append(tm)


# Ntemplate = 200

# plt.figure()
# plt.imshow(templates[Ntemplate].T)
# plt.gca().invert_yaxis()
# plt.title('Template %d' % Ntemplate)
# plt.savefig('Template_%d.png' % Ntemplate)
# plt.show()

print('Loading data spectrum')
spectrograms = np.load('/extra/scratch03/jantoran/Documents/moby_dick/template_boosting_solution/data/processed_data_spectrum_250.npy')

spectrograms -= spectrograms.mean(axis=(1,2), keepdims=True)
spectrograms /= spectrograms.std(axis=(1,2), keepdims=True)

print('spectrograms loaded and normalized, shape:', spectrograms.shape)
# spec = spectograms[7]
# xcorr = signal.correlate2d(spec, templates[Ntemplate], mode='full', boundary='fill', fillvalue=0)
#
# plt.figure()
# plt.imshow(xcorr.T)
# plt.gca().invert_yaxis()
# plt.title('Cross correlation of spectogram with template %d' % Ntemplate)
# plt.savefig('xcorr_%d.png' % Ntemplate)
# plt.show()

chunk_n = 1
for chunk_index in range(0,spectrograms.shape[0],chunk):
    np.save('/extra/scratch03/jantoran/Documents/moby_dick/template_boosting_solution/data/spectrograms/processed_data_norm_spectrum_250_%d.npy' % (chunk_n,), spectrograms[chunk_index:chunk_index+chunk])
    print('Saved chunk number %s/%s' % (chunk_n,int(spectrograms.shape[0]/chunk)))
    chunk_n = chunk_n + 1

