# main
import numpy as np
from customized_cnn_loss import CycleGAN
from parameter_handle import parameter
from util import ImageGenerator

if __name__ == '__main__':
    para = parameter()
    para.pic_dir = "./face_person"
    cycleGAN = CycleGAN(para)

    IG_A = ImageGenerator(root=r'C:\PR_project\split_data_actor\1',
                resize=para.resize, crop=para.crop)
    IG_B = ImageGenerator(root=r'C:\PR_project\split_data_actor\2',
                resize=para.resize, crop=para.crop)

    cycleGAN.fit(IG_A, IG_B)