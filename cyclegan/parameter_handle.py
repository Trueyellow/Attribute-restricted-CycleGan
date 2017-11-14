# class for parameter handling
import numpy as np
from pprint import pprint


class parameter(object):
    def __init__(self,
                 DATA_ROOT='',  # path to images (should have subfolders 'train', 'val', etc)
                 shapeA=(128, 128, 3),
                 shapeB=(128, 128, 3),
                 label_num=8,
                 label_shape=(128, 128, 8),
                 resize=(143, 143),  # scale images to this size
                 crop=(128, 128),  # then crop to this size
                 # net definition
                 use_lsgan=1,  # if 1, use least square GAN, if 0, use vanilla GAN
                 ngf=64,  # of gen filters in first conv layer
                 ndf=64,  # of discrim filters in first conv layer
                 lmbd=10.0,
                 idloss=0,

                 # optimizers
                 lr_D=0.0001,  # initial learning rate for adam
                 lr_G=0.0002,
                 beta1=0.5,  # momentum term of adam

                 # training parameters
                 batch_size=1,  # images in batch
                 niter=100000,  # of iter at starting learning rate
                 pool_size=50,  # the size of image buffer that stores previously generated images
                 save_iter=500,
                 d_iter=1,

                 # dirs
                 pic_dir='./without_id',
                 niter_decay=100,  # of iter to linearly decay learning rate to zero
                 ntrain=np.inf,  # of examples per epoch. np.inf for full dataset
                 flip=1,  # if flip the images for data argumentation
                 display_id=10,  # display window id.
                 display_winsize=128,  # display window size
                 display_freq=25,  # display the current results every display_freq iterations
                 name='',  # name of the experiment, should generally be passed on the command line
                 ):

        assert shapeA[0:2] == crop
        self.__dict__.update(locals())

    def summary(self):
        pprint(self.__dict__)
