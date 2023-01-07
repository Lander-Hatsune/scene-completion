# complete.py <src_img> <src_mask> <candidate_dir>
import os
import cv2
import sys
import time
import numpy as np
import igraph as ig
from PIL import Image
from queue import Queue
import matplotlib.pyplot as plt
from scipy.signal import convolve
from scipy.ndimage import laplace
from joblib import Parallel, delayed

BORDER_RANGE = 80

class MaskedImg:
    
    def __init__(self, img, mask):
        '''
        img: (C, H, W),
        mask: (H, W), 0 to replace, 1 to keep
        '''
        img = img.astype(np.int64)
        mask = mask.astype(np.int64)
        
        assert (0 <= img).all() and (img < 256).all() and img.shape[0] == 3
        assert np.logical_or(mask == 0, mask == 1).all()
        
        self._img = img # C, H, W
        self._mask = mask # H, W
        assert(self._img.shape[1:] == self._mask.shape)
        self.shape = self._mask.shape
        dilated_mask = cv2.erode(
            self._mask.astype(np.uint8),
            np.ones((BORDER_RANGE * 2, BORDER_RANGE * 2)) # kernel
        )
        self._border_mask = dilated_mask ^ self._mask
        
    def border(self, img=None):
        # return 'border' part
        if img is None:
            img = self._img
        if self._border_mask.shape != img.shape[1:]: # cut to minimal same shape
            minshape = np.minimum(self._border_mask.shape, img.shape[1:])
            border_mask = self._border_mask[:minshape[0], :minshape[1]]
            img = img[:, :minshape[0], :minshape[1]]
        else:
            border_mask = self._border_mask
        return img * border_mask

    def mask(self, img=None):
        # return 'keep' part
        if img is None:
            img = self._img
        if self._mask.shape != img.shape: # cut to minimal same shape
            minshape = np.minimum(self._mask.shape, img.shape[1:])
            mask = self._mask[:minshape[0], :minshape[1]]
            img = img[:, :minshape[0], :minshape[1]]
        else:
            mask = self._mask
        return img * mask

    def invmask(self, img=None):
        # return 'replace' part
        if img is None:
            img = self._img
        if self._mask.shape != img.shape: # cut to minimal same shape
            minshape = np.minimum(self._mask.shape, img.shape[1:])
            mask = self._mask[:minshape[0], :minshape[1]]
            img = img[:, :minshape[0], :minshape[1]]
        else:
            mask = self._mask
        return img * (1 - mask)

    def match(self, candi_img):
        '''
        candi_img: shape (C, H, W)
        return: patch ('replace' part + 'border' part)
        '''
        print('--- Matching candi_img to src.border ---')
        assert (0 <= candi_img).all() and \
            (candi_img < 256).all() and \
            candi_img.shape[0] == 3
        candi_img = candi_img.astype(np.int64)
        
        X = self.border()
        X2 = (X ** 2).sum()
        
        candi_img_pad = np.pad(candi_img, ((0,),
                                           (self.shape[0] - 1,),
                                           (self.shape[1] - 1,)))
                                           
        allZ2 = candi_img_pad ** 2

        masksumZ2 = convolve(
            allZ2.transpose((1, 2, 0)),
            np.expand_dims(self._border_mask[::-1, ::-1], 2),
            mode='valid'
        ).transpose((2, 0, 1)).sum(0)

        allXZ = convolve(
            candi_img_pad,
            X[::-1, ::-1, ::-1],
            mode='valid'
        ).squeeze()

        min_dis = np.inf
        chosen_pos = None

        time_start = time.time()

        def _getdis(cx, cy):
            cx, cy = cx * 10, cy * 10
            assert(candi_img_pad[:, cx:, cy:].shape[1:] >= self.shape)
            Z = self.border(candi_img_pad[:, cx:, cy:])

            XZ = allXZ[cx, cy]
            assert (XZ == (X * Z).sum()), \
                f'XZ: {XZ}, XZ_brute: {(X * Z).sum()}'

            Z2 = masksumZ2[cx, cy]
            assert (Z2 == (Z ** 2).sum()), \
                f'Z2: {Z2}, Z2_brute: {(Z ** 2).sum()}'

            dis = (X2 + Z2 - 2 * XZ)
            # dis = Z2 - 2 * XZ
            assert (dis == ((X - Z) ** 2).sum()), \
                f'dis: {dis}, dis_brute: {((X - Z) ** 2).sum()}'

            # dis = ((X - Z) ** 2).sum() # 26.97s
            if (cx % 100 == 0 and cy == 0):
                print(f'[{cx}/{masksumZ2.shape[0]}]')
            return dis

        dilated_shape = (masksumZ2.shape[0] // 10, masksumZ2.shape[1] // 10)
        dismap = np.array(
            Parallel(n_jobs=12)(delayed(_getdis)(cx, cy) \
                                for cx, cy in np.ndindex(dilated_shape))) # 105s / 45s
        print('minimum dis:', dismap.min())
        chosen_pos = np.unravel_index(np.argmin(dismap), dilated_shape)
        chosen_pos = chosen_pos[0] * 10, chosen_pos[1] * 10
        print(chosen_pos)

        print(f'elapsed: {time.time() - time_start:.3f}s')
        return self.invmask(candi_img_pad[:, chosen_pos[0]:, chosen_pos[1]:]) + \
            self.border(candi_img_pad[:, chosen_pos[0]:, chosen_pos[1]:])

    def cut_patch(self, patch):
        '''
        cut border part to (A: keep, B: replace) part
        
        patch: shape (C, H, W), 'replace' part + 'border' part
        return: patch cut, ready to substitute
        '''
        print('cutting patch')
        sname = self.shape[0] * self.shape[1]
        tname = sname + 1
        edges = []

        inf = 1024

        # add edges downward
        for mx, my in np.ndindex((self.shape[0] - 1, self.shape[1])):
            pix_name = mx * self.shape[1] + my
            pix_down_name = (mx + 1) * self.shape[1] + my

            # s-border
            if (not self._border_mask[mx, my] and self._mask[mx, my]) and \
               self._border_mask[mx + 1, my]:
                edges.append((sname, pix_down_name, inf))

            # border-s
            if self._border_mask[mx, my] and \
               (not self._border_mask[mx + 1, my] and self._mask[mx + 1, my]):
                edges.append((pix_name, sname, inf))

            # border-border
            if self._border_mask[mx, my] and \
               self._border_mask[mx + 1, my]:
                w = abs(self._img[:, mx, my] - patch[:, mx, my]).sum() \
                    + abs(self._img[:, mx + 1, my] - patch[:, mx + 1, my]).sum()
                edges.append((pix_name, pix_down_name, w))
                
            # border-t
            if self._border_mask[mx, my] and \
               not self._mask[mx + 1, my]:
                edges.append((pix_name, tname, inf))

            # t-border
            if not self._mask[mx, my] and \
               self._border_mask[mx + 1, my]:
                edges.append((tname, pix_down_name, inf))

        # add edges rightward
        for mx, my in np.ndindex((self.shape[0], self.shape[1] - 1)):
            pix_name = mx * self.shape[1] + my
            pix_right_name = mx * self.shape[1] + my + 1

            # s-border
            if (not self._border_mask[mx, my] and self._mask[mx, my]) and \
               self._border_mask[mx, my + 1]:
                edges.append((sname, pix_right_name, inf))

            # border-s
            if self._border_mask[mx, my] and \
               (not self._border_mask[mx, my + 1] and self._mask[mx, my + 1]):
                edges.append((pix_name, sname, inf))

            # border-border
            if self._border_mask[mx, my] and \
               self._border_mask[mx, my + 1]:
                w = abs(self._img[:, mx, my] - patch[:, mx, my]).sum() \
                    + abs(self._img[:, mx, my + 1] - patch[:, mx, my + 1]).sum()
                edges.append((pix_name, pix_right_name, w))
                
            # border-t
            if self._border_mask[mx, my] and \
               not self._mask[mx, my + 1]:
                edges.append((pix_name, tname, inf))

            # t-border
            if not self._mask[mx, my] and \
               self._border_mask[mx, my + 1]:
                edges.append((tname, pix_right_name, inf))

        edges = eval(str(edges)) # cannot do mincut with weights if deleted(???)
            
        graph = ig.Graph.TupleList(edges, weights=True)
        ig.summary(graph)

        mincut = graph.mincut(
            graph.vs.find(name=sname), graph.vs.find(name=tname),
            capacity='weight'
        )
        print(mincut)
        _, B_ids = mincut.partition

        def ids2mask(ids):
            mask = np.zeros(self.shape)
            for vid in ids: # vertex id
                name = graph.vs.find(vid)['name']
                if name >= sname:
                    continue
                x, y = name // self.shape[1], name % self.shape[1]
                mask[x, y] = 1
            return mask

        Bmask = ids2mask(B_ids)
        return Bmask

    # https://github.com/ar90n/poisson-blending-in-5lines
    def blend(self, patch):
        print('--- Blending ---')

        result_n = (patch.invmask(self._img) + patch.mask()) / 255.0
        patch_n = patch._img / 255.0
        mask = patch._mask

        time_start = time.time()
        def _iter_channel(result_n, patch_n):
            for i in range(8192):
                result_n = result_n + 0.25 * mask * \
                    laplace(result_n - patch_n)
            return result_n
        result_n = np.array(
            Parallel(n_jobs=3)(delayed(_iter_channel)(result_n[c], patch_n[c]) \
                               for c in range(3)))
        print(f'elapsed: {time.time() - time_start:.3f}s')
        return result_n.clip(0, 1) * 255

if __name__ == '__main__':

    # images use C, H, W axes, masks use H, W axes
    src_img = np.array(Image.open(sys.argv[1]))[..., :3].transpose((2, 0, 1))
    src_mask = (np.array(Image.open(sys.argv[2]))[..., 0] >= 128).astype(np.int64)
    src = MaskedImg(src_img, src_mask)

    candi_dir = sys.argv[3]

    # Image.fromarray(src.border().transpose((1, 2, 0))
    #                 .astype(np.uint8)).show()
    
    for candi_name in os.listdir(candi_dir):
        print(f'\nCandidate image: {candi_name}')
        candi_path = os.path.join(candi_dir, candi_name)
        candi_img = np.array(Image.open(candi_path))[..., :3].transpose((2, 0, 1))

        patch_img = src.match(candi_img)
        # Image.fromarray(patch_img.transpose((1, 2, 0))
        #                 .astype(np.uint8)).show()

        patch_mask = np.logical_or(src.cut_patch(patch_img), (1 - src_mask))
        # Image.fromarray((patch_mask * 255).astype(np.uint8)).show()
        
        patch = MaskedImg(patch_img, patch_mask)
        # Image.fromarray((patch.mask() + patch.invmask(src_img))
        #                 .transpose((1, 2, 0)).astype(np.uint8)).show()

        result = src.blend(patch)
        # Image.fromarray(result.transpose((1, 2, 0)).astype(np.uint8)).show()

        os.system(f'mkdir -p results/{candi_dir}')
        (Image.fromarray(result.transpose((1, 2, 0)).astype(np.uint8))
         .save(f'results/{candi_path}'))
        print(f'results/{candi_path} saved')
    
