import os
import numpy as np
from scipy.misc import imsave, imread, imresize

# for data visualization and save data
def vis_grid(X, nh, nw, save_path=None):

    h, w = X.shape[1:3]
    img = np.zeros((h*nh, w*nw, 3))
    for n, x in enumerate(X):
        j = int(n/nw)
        i = n % nw
        if n >= nh*nw: break
        img[j*h:j*h+h, i*w:i*w+w, :] = x

    if save_path is not None:
        imsave(save_path, img)
    return img


class ImageGenerator(object):
    def __init__(self, root, resize=None, crop=None, flip=None):
        self.img_list = os.listdir(root)
        self.root = root
        self.resize = resize
        self.crop = crop
        self.flip = flip

        print('ImageGenerator from {} [{}]'.format(root, len(self.img_list)))

    def __call__(self, bs, return_label=False):
        while True:
            try:
                imgs = []
                labels = []
                for _ in range(bs):
                    random_choiced = np.random.choice(self.img_list)
                    img = imread(os.path.join(self.root, random_choiced))
                    print(os.path.join(self.root, random_choiced))
                    label = int(random_choiced.split("_")[0]) - 1
                    if self.resize:
                        img = imresize(img, self.resize)
                    if self.crop:
                        left = np.random.randint(0, img.shape[0]-self.crop[0])
                        top  = np.random.randint(0, img.shape[1]-self.crop[1])
                        img = img[left:left+self.crop[0], top:top+self.crop[1]]
                    if self.flip:
                        if np.random.random() > 0.5:
                            img = img[:, ::-1, :]
                    labels.append(label)
                    imgs.append(img)

                imgs = np.array(imgs)
                imgs = imgs/127.5-1

                if return_label:
                    return imgs, np.array(label)
                else:
                    return imgs
            except:
                pass