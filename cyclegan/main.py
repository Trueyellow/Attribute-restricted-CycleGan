# main
import numpy as np
from cyclegan.customized_cnn_loss import CycleGAN
from cyclegan.parameter_handle import parameter
from cyclegan.util import ImageGenerator

if __name__ == '__main__':
    para = parameter()
    para.pic_dir = "./face_person"
    cycleGAN = CycleGAN(para)

    IG_A = ImageGenerator(root=r'E:\DATASET\pr\split_data_actor\1',
                resize=para.resize, crop=para.crop)
    IG_B = ImageGenerator(root=r'E:\DATASET\pr\split_data_actor\2',
                resize=para.resize, crop=para.crop)

    cycleGAN.fit(IG_A, IG_B)