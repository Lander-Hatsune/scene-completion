# complete.py <src_img> <src_mask> <candidate_dir>
import os
import cv2
import sys
import logging as log
import numpy as np
from PIL import Image
from queue import Queue
import matplotlib.pyplot as plt

BORDER_RANGE = 80

class MaskedImg:
    
    def __init__(self, img, mask):
        self._img = img
        self._mask = mask
        dilated_mask = cv2.erode(
            self._mask,
            np.ones((BORDER_RANGE * 2, BORDER_RANGE * 2)) # kernel
        )
        self._border_mask = dilated_mask ^ self._mask
        
    def border(self, img=None):
        if img is None:
            img = self._img
        if self._border_mask.shape != img.shape: # cut to minimal same shape
            minshape = np.minimum(self._border_mask.shape, img.shape[:2])
            border_mask = self._border_mask[:minshape[0], :minshape[1]]
            img = img[:minshape[0], :minshape[1]]
        return img * np.expand_dims(border_mask, 2)

    def mask(self, img=None):
        if img is None:
            img = self._img
        if self._mask.shape != img.shape: # cut to minimal same shape
            minshape = np.minimum(self._mask.shape, img.shape[:2])
            mask = self._mask[:minshape[0], :minshape[1]]
            img = img[:minshape[0], :minshape[1]]
        return img * np.expand_dims(mask, 2)

    def invmask(self, img=None):
        if img is None:
            img = self._img
        if self._mask.shape != img.shape: # cut to minimal same shape
            minshape = np.minimum(self._mask.shape, img.shape[:2])
            mask = self._mask[:minshape[0], :minshape[1]]
            img = img[:minshape[0], :minshape[1]]
        return img * np.expand_dims(1 - mask, 2)

    def match(self, candi_img):
        candi_img_pad = np.pad(candi_img,
                               (*np.expand_dims(self._mask.shape, 1), (0,)))
        X = self.border()
        X2 = (X ** 2).sum((0, 1))
        
        allZ2 = candi_img ** 2
        masksumZ2 = cv2.filter2D(
            np.pad(allZ2, ((self._mask.shape[0] - 1, 0),
                           (self._mask.shape[1] - 1, 0),
                           (0, 0))),
            -1,
            self._mask,
            anchor=(0, 0),
            borderType=cv2.BORDER_CONSTANT
        )

        min_dis = np.inf
        chosen_pos = None
        for cx, cy in np.ndindex(
                *np.subtract(candi_img_pad.shape[:2], self._mask.shape)):
            Z = self.border(candi_img_pad[cx:, cy:])

            XZ = cv2.filter2D(
                Z, -1, X,
                anchor=(0, 0),
                borderType=cv2.BORDER_CONSTANT
            )[0, 0]
            
            Z2 = masksumZ2[cx, cy]

            dis = (X2 + Z2 - 2 * XZ).sum()
            if dis < min_dis:
                min_dis = dis
                chosen_pos = (cx, cy)
            break

        return chosen_pos
        
src_img = np.array(Image.open(sys.argv[1]))[..., :3]
src_mask = np.array(Image.open(sys.argv[2]))[..., 0] // 255
src = MaskedImg(src_img, src_mask)

candi_dir = sys.argv[3]

if __name__ == '__main__':

    for candi_name in os.listdir(candi_dir):
        log.info(f'Candidate image: {candi_name}')
        candi_path = os.path.join(candi_dir, candi_name)
        candi_img = np.array(Image.open(candi_path))[..., :3]
        chosen_pos = src.match(candi_img)

        patch = src.invmask(candi_img[chosen_pos[0]:, chosen_pos[1]:])
    
