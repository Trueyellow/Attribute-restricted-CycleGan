# Cycle gan based on pytorch version of https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/cycle_gan_model.py
#

import os
import keras.backend as K
from .build_DG import basic_D, resnet_6blocks, basic_D_A, resnet_6blocks_A
from keras.layers import Input
from keras.optimizers import Adam
from keras.models import Model
from .loss_function import loss_fn
import numpy as np
import sys
from .util import vis_grid

defineG = resnet_6blocks
defineD = basic_D
defineG_A = resnet_6blocks_A
defineD_A = basic_D_A


class CycleGAN():
    def __init__(self, opt):

        gen_B = defineG(opt.shapeA,  opt.shapeB[2], ngf=opt.ngf, name='gen_B')

        dis_B = basic_D(opt.shapeB,  opt.ndf, use_sigmoid=not opt.use_lsgan,
                        name='dis_B')

        gen_A = defineG_A(opt.shapeB, opt.label_shape, opt.shapeB[2], ngf=opt.ngf,
                        name='gen_A')

        dis_A = defineD_A(opt.shapeA, opt.label_shape, opt.ndf, use_sigmoid=not opt.use_lsgan,
                        name='dis_A')

        self.init_network(gen_B)
        self.init_network(dis_B)
        self.init_network(gen_A)
        self.init_network(dis_A)

        # building loss function

        # real image input
        real_A = Input(opt.shapeA)
        real_B = Input(opt.shapeB)
        true_label = Input(opt.label_shape)
        fake_label = Input(opt.label_shape)
        # input from fake image pool
        fake_A_pool = Input(opt.shapeA)
        fake_B_pool = Input(opt.shapeB)


        fake_B = gen_B(real_A)
        rec_A = gen_A(fake_B)  # = gen_A(gen_B(real_A))
        fake_A = gen_A(real_B)
        rec_B = gen_B(fake_A)  # = gen_B(gen_A(real_B))

        # discriminator A function output
        dis_A_real = dis_A(real_A)
        dis_A_fake = dis_A(fake_A_pool)
        Gdis_A = dis_A(fake_A)

        # discriminator B function output
        dis_B_real = dis_B(real_B)
        dis_B_fake = dis_B(fake_B_pool)
        Gdis_B = dis_A(fake_A)

        # DA, GA loss
        loss_DA_real = loss_fn(dis_A_real, K.ones_like(dis_A_real))
        loss_DA_fake = loss_fn(dis_A_fake, K.zeros_like(dis_A_fake))
        loss_DA = loss_DA_real + loss_DA_fake
        # real A with correct label
        loss_GA = loss_fn(Gdis_A, K.zeros_like(Gdis_A))
        loss_GA_wrong = loss_fn(Gdis_A, K.ones_like(Gdis_A))
        loss_cycA = K.mean(K.abs(rec_A - real_A))

        # DB, GB loss
        loss_DB_real = loss_fn(dis_B_real, K.ones_like(dis_B_real))
        loss_DB_fake = loss_fn(dis_B_fake, K.zeros_like(dis_B_fake))
        loss_DB = loss_DB_real + loss_DB_fake
        loss_GB = loss_fn(Gdis_B, K.zeros_like(Gdis_B))
        loss_cycB = K.mean(K.abs(rec_B - real_B))

        # cycle loss
        loss_cyc = loss_cycA + loss_cycB

        # D's total loss
        loss_D = loss_DA + loss_DB

        # G's total loss
        loss_G = loss_GA + loss_GB + opt.lmbd * loss_cyc

        weightsD = dis_A.trainable_weights + dis_B.trainable_weights
        weightsG = gen_A.trainable_weights + gen_B.trainable_weights

        # training function for discriminator
        # update both of D_A, D_B based on the total loss of dis_a, dis_b
        training_updates = Adam(lr=opt.lr_D, beta_1=0.5).get_updates(loss_D, weightsD)
        netD_train = K.function([real_A, real_B, fake_A_pool, fake_B_pool], [loss_DA / 2, loss_DB / 2], training_updates)

        # training function for generator
        # update both of D_A, D_B based on the total loss of GA, GB and CYCLE loss
        training_updates = Adam(lr=opt.lr_G, beta_1=0.5).get_updates(loss_G, weightsG)
        netG_train = K.function([real_A, real_B], [loss_GA, loss_GB, loss_cyc], training_updates)

        self.G_trainner = netG_train
        self.D_trainner = netD_train
        self.AtoB = gen_B
        self.BtoA = gen_A
        self.DisA = dis_A
        self.DisB = dis_B
        self.opt = opt

    def fit(self, img_A_generator, img_B_generator):
        opt = self.opt
        if not os.path.exists(opt.pic_dir):
            os.mkdir(opt.pic_dir)

        bs = opt.batch_size

        fake_A_pool = []
        fake_B_pool = []

        iteration = 0

        if os.path.exists(os.path.join(opt.pic_dir, 'a2b.h5')):
            self.AtoB.load_weights(os.path.join(opt.pic_dir, 'a2b.h5'))
            self.BtoA.load_weights(os.path.join(opt.pic_dir, 'b2a.h5'))

        while iteration < opt.niter:
            print('iteration: {}'.format(iteration))
            # sample
            real_A = img_A_generator(bs)
            real_B = img_B_generator(bs)

            # fake pool
            try:
                fake_A_pool.extend(self.BtoA.predict(real_B))
                fake_B_pool.extend(self.AtoB.predict(real_A))
                fake_A_pool = fake_A_pool[-opt.pool_size:]
                fake_B_pool = fake_B_pool[-opt.pool_size:]

                fake_A = [fake_A_pool[ind] for ind in np.random.choice(len(fake_A_pool), size=(bs,), replace=False)]
                fake_B = [fake_B_pool[ind] for ind in np.random.choice(len(fake_B_pool), size=(bs,), replace=False)]
                fake_A = np.array(fake_A)
                fake_B = np.array(fake_B)

                # ones = np.ones((bs,) + self.G_trainner.output_shape[0][1:])
                # zeros = np.zeros((bs,) + self.G_trainner.output_shape[0][1:])

                # train
                for _ in range(opt.d_iter):
                    errDA, errDB = self.D_trainner([real_A, real_B, fake_A, fake_B])

                errGA, errGB, errCyc = self.G_trainner([real_A, real_B])
            except:
                continue


            print('Generator Loss:')
            print('GA: {}  | GB: {} || G_cycle: {}\n'.format(np.mean(errGA), np.mean(errGB), errCyc))

            print('Discriminator Loss:')
            print('D_A: {} | D_B: {}\n'.format(np.mean(errDA), np.mean(errDB)))

            print("Dis_A")
            res = self.DisA.predict(real_A)
            print("real_A: {}".format(res.mean()))
            res = self.DisA.predict(fake_A)
            print("fake_A: {}\n\n".format(res.mean()))

            if iteration % opt.save_iter == 0:
                imga = real_A
                imga2b = self.AtoB.predict(imga)
                imga2b2a = self.BtoA.predict(imga2b)

                imgb = real_B
                imgb2a = self.BtoA.predict(imgb)
                imgb2a2b = self.AtoB.predict(imgb2a)

                vis_grid(np.concatenate([imga, imga2b, imga2b2a, imgb, imgb2a, imgb2a2b],
                                        axis=0),
                         6, bs, os.path.join(opt.pic_dir, '{}.png'.format(iteration)))

                self.AtoB.save(os.path.join(opt.pic_dir, 'a2b.h5'))
                self.BtoA.save(os.path.join(opt.pic_dir, 'b2a.h5'))

            # import ipdb
            #               ipdb.set_trace()
            iteration += 1
            sys.stdout.flush()

    @staticmethod
    def init_network(model):
        for w in model.weights:
            if w.name.startswith('conv2d') and w.name.endswith('kernel'):
                value = np.random.normal(loc=0.0, scale=0.02, size=w.get_value().shape)
                w.set_value(value.astype('float32'))
            if w.name.startswith('conv2d') and w.name.endswith('bias'):
                value = np.zeros(w.get_value().shape)
                w.set_value(value.astype('float32'))