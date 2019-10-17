from skimage.io import imread
from sift import SIFT
import matplotlib.pyplot as plt

import pickle
from os.path import isfile

num_img = 3
img_num_offset = 3
kp_pyrs = []
ims = []

for j in range(1, num_img + 1):
    imageName = 'IMG_115' + str(j + img_num_offset) + '.jpg'
    im = imread('images/' + imageName)

    ims.append(im)

    if isfile('results/kp_pyr%s.pkl' % str(j + img_num_offset)):
        kp_pyrs.append(pickle.load(open('results/kp_pyr%s.pkl' % str(j + img_num_offset), 'rb')))
        continue

    print('Performing SIFT on image %s' % imageName)

    sift_detector = SIFT(im)
    _ = sift_detector.get_features()
    kp_pyrs.append(sift_detector.kp_pyr)

    pickle.dump(sift_detector.kp_pyr, open('results/kp_pyr%s.pkl' % str(j + img_num_offset), 'wb'))
    pickle.dump(sift_detector.feats, open('results/feat_pyr%s.pkl' % str(j + img_num_offset), 'wb'))

for i in range(len(kp_pyrs[0])):
    _, ax = plt.subplots(1, 3)
    ax[0].imshow(ims[0])

    kps = kp_pyrs[0][i] * (2 ** i)
    ax[0].scatter(kps[:, 0], kps[:, 1], c='b', s=2.5)

    ax[1].imshow(ims[1])

    kps = kp_pyrs[1][i] * (2 ** i)
    ax[1].scatter(kps[:, 0], kps[:, 1], c='b', s=2.5)

    ax[2].imshow(ims[2])

    kps = kp_pyrs[2][i] * (2 ** i)
    ax[2].scatter(kps[:, 0], kps[:, 1], c='b', s=2.5)

    plt.show()
