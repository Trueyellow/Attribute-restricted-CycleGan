# main
import numpy as np
from cyclegan.custom_loss import CycleGAN
from cyclegan.parameter_handle import parameter
from cyclegan.util import ImageGenerator

if __name__ == '__main__':
    para = parameter()
    para.pic_dir = "./face"
    cycleGAN = CycleGAN(para)

    IG_A = ImageGenerator(root=r'E:\DATASET\pr\split_data_img\angry',
                resize=para.resize, crop=para.crop)
    IG_B = ImageGenerator(root=r'E:\DATASET\pr\split_data_img\happy',
                resize=para.resize, crop=para.crop)

    cycleGAN.fit(IG_A, IG_B)