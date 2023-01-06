# complete.py <src_img> <src_mask> <candidate_dir>
import os
import cv2
import sys
import numpy as np
# import logging as log
from PIL import Image
from queue import Queue
from scipy.signal import convolve
import matplotlib.pyplot as plt
# from joblib import Parallel, delayed

BORDER_RANGE = 20

class MaskedImg:
    
    def __init__(self, img, mask):

        # debug
        # img = np.random.randint(0, 256, size=(3, 5, 5))
        # mask = np.random.randint(0, 2, size=(5, 5)).astype(np.uint8)
        # print('img:', img)
        # print('mask:', mask)

        img = img.astype(np.int64)
        mask = mask.astype(np.int64)
        
        assert (0 <= img).all() and (img < 256).all() and img.shape[0] == 3
        assert np.logical_or(mask == 0, mask == 1).all()
        
        self._img = img # C, H, W
        self._mask = mask
        assert(self._img.shape[1:] == self._mask.shape)
        self.shape = self._mask.shape
        dilated_mask = cv2.erode(
            self._mask.astype(np.uint8),
            np.ones((BORDER_RANGE * 2, BORDER_RANGE * 2)) # kernel
        )
        self._border_mask = dilated_mask ^ self._mask
        
    def border(self, img=None):
        if img is None:
            img = self._img
        if self._border_mask.shape != img.shape[1:]: # cut to minimal same shape
            minshape = np.minimum(self._border_mask.shape, img.shape[1:])
            border_mask = self._border_mask[:minshape[0], :minshape[1]]
            img = img[:, :minshape[0], :minshape[1]]
            # print(f'cut to same shape: {minshape}')
        else:
            border_mask = self._border_mask
        return img * border_mask

    def mask(self, img=None):
        if img is None:
            img = self._img
        if self._mask.shape != img.shape: # cut to minimal same shape
            minshape = np.minimum(self._mask.shape, img.shape[1:])
            mask = self._mask[:minshape[0], :minshape[1]]
            img = img[:, :minshape[0], :minshape[1]]
            # print(f'cut to same shape: {minshape}')
        else:
            mask = self._mask
        return img * mask

    def invmask(self, img=None):
        if img is None:
            img = self._img
        if self._mask.shape != img.shape: # cut to minimal same shape
            minshape = np.minimum(self._mask.shape, img.shape[1:])
            mask = self._mask[:minshape[0], :minshape[1]]
            img = img[:, :minshape[0], :minshape[1]]
            # print(f'cut to same shape: {minshape}')
        else:
            mask = self._mask
        return img * (1 - mask)

    def match(self, candi_img):

        # candi_img = np.random.randint(0, 256, size=(3, 8, 8)) # debug
        assert (0 <= candi_img).all() and \
            (candi_img < 256).all() and \
            candi_img.shape[0] == 3
        candi_img = candi_img.astype(np.int64)
        
        X = self.border()
        # print('X.shape:', X.shape)
        X2 = (X ** 2).sum((1, 2))
        # print('X2:', X2)
        
        candi_img_pad = np.pad(candi_img, ((0,),
                                           (self.shape[0] - 1,),
                                           (self.shape[1] - 1,)))
                                           
        # print('candi_img_pad.shape:', candi_img_pad.shape)
        allZ2 = candi_img_pad ** 2
        # print('allZ2.shape:', allZ2.shape)

        masksumZ2 = convolve(
            allZ2.transpose((1, 2, 0)),
            np.expand_dims(self._border_mask[::-1, ::-1], 2),
            mode='valid'
        ).transpose((2, 0, 1))
        # print('masksumZ2.shape:', masksumZ2.shape)

        min_dis = np.inf
        chosen_pos = None
        for cx, cy in np.ndindex(masksumZ2.shape[1:]):
            assert(candi_img_pad[:, cx:, cy:].shape[1:] >= self.shape)
            Z = self.border(candi_img_pad[:, cx:, cy:])
            # print('Z.shape:', Z.shape)

            XZ = (X * Z).sum((1, 2))
            # print('XZ:', XZ)

            # XZ_ = np.concatenate((
            #     convolve(X[..., 0], Z[::-1, ::-1, 0], mode='valid'),
            #     convolve(X[..., 1], Z[::-1, ::-1, 1], mode='valid'),
            #     convolve(X[..., 2], Z[::-1, ::-1, 2], mode='valid'),
            # )).squeeze()

            Z2 = masksumZ2[:, cx, cy]
            assert (Z2 == (Z ** 2).sum((1, 2))).all(), \
                f'Z2: {Z2}, Z2_brute: {(Z ** 2).sum((1, 2))}'

            dis = (X2 + Z2 - 2 * XZ)
            dis_brute = ((X - Z) ** 2).sum((1, 2))
            
            assert (dis == dis_brute).all(), \
                f'dis: {dis}, dis_brute: {dis_brute}'
            
            if dis.sum() < min_dis:
                min_dis = dis.sum()
                chosen_pos = (cx, cy)
                print(cx, cy, min_dis)

        return self.invmask(candi_img_pad[:, chosen_pos[0]:, chosen_pos[1]:])

if __name__ == '__main__':

    # images use C, H, W axes, masks use H, W axes
    src_img = np.array(Image.open(sys.argv[1]))[..., :3].transpose((2, 0, 1))
    src_mask = (np.array(Image.open(sys.argv[2]))[..., 0] >= 128).astype(np.int64)
    src = MaskedImg(src_img, src_mask)

    candi_dir = sys.argv[3]

    Image.fromarray(src.border().transpose((1, 2, 0)).astype(np.uint8)).show('border(X)')
    
    for candi_name in os.listdir(candi_dir):
        print(f'Candidate image: {candi_name}')
        candi_path = os.path.join(candi_dir, candi_name)
        candi_img = np.array(Image.open(candi_path))[..., :3].transpose((2, 0, 1))
        patch = src.match(candi_img).transpose((1, 2, 0)).astype(np.uint8)
        print(patch.shape)
        Image.fromarray(patch.astype(np.uint8)).show(f'{candi_name} chosen patch')

        
    
