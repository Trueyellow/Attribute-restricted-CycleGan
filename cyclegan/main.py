# main
import numpy as np
from custom_loss import CycleGAN
from parameter_handle import parameter
from util import ImageGenerator

if __name__ == '__main__':
    para = parameter()
    cycleGAN = CycleGAN(para)

    # IG_A = ImageGenerator(root=r'F:\DATASET\horse2zebra\trainA',
    #             resize=para.resize, crop=para.crop)
    # IG_B = ImageGenerator(root=r'F:\DATASET\horse2zebra\trainB',
    #             resize=para.resize, crop=para.crop)
    #
    # cycleGAN.fit(IG_A, IG_B)