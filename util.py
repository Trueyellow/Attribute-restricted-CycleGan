# Pattern Recognition final project Group 8 Kaixiang Huang
# cycle_GAN util
import os
import numpy as np
from scipy.misc import imsave, imread, imresize


def label_generate(label, label_num, label_shape_G, label_shape_D):
    true_label = label
    fake_label = np.random.choice(np.delete(np.arange(label_num), true_label, axis=0))
    label_G_true = np.zeros(shape=label_shape_G, dtype=float)
    label_D_true = np.zeros(shape=label_shape_D, dtype=float)
    label_D_fake = np.zeros(shape=label_shape_D, dtype=float)
    label_G_true[:, :, true_label] = 1
    label_D_true[:, :, true_label] = 1
    label_D_fake[:, :, fake_label] = 1
    label_G_true = label_G_true[np.newaxis, :, :, :]
    label_D_true = label_D_true[np.newaxis, :, :, :]
    label_D_fake = label_D_fake[np.newaxis, :, :, :]
    return label_G_true, label_D_true, label_D_fake


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
        self.img_list.sort(key=lambda x: int(x[:-4]))
        self.root = root
        self.resize = resize
        self.crop = crop
        self.flip = flip

        print('ImageGenerator from {} [{}]'.format(root, len(self.img_list)))

    def __call__(self, bs, return_label=False, random=True):
        while True:
            try:
                imgs = []
                labels = []
                if random:
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
                else:
                    for image_path in self.img_list:
                        img = imread(os.path.join(self.root, image_path))
                        if self.resize:
                            img = imresize(img, self.resize)
                        if self.crop:
                            left = np.random.randint(0, img.shape[0] - self.crop[0])
                            top = np.random.randint(0, img.shape[1] - self.crop[1])
                            img = img[left:left + self.crop[0], top:top + self.crop[1]]
                        if self.flip:
                            if np.random.random() > 0.5:
                                img = img[:, ::-1, :]
                        imgs.append(img)

                imgs = np.array(imgs)
                imgs = imgs/127.5-1

                if return_label:
                    return imgs, np.array(label)
                else:
                    return imgs
            except:
                pass