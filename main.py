# Pattern Recognition final project Group 8 Kaixiang Huang
# Training main function
import os
from parameter_handle import parameter
from DEMO_cycleGAN import CycleGAN
from util import ImageGenerator

if __name__ == '__main__':
    para = parameter()
    para.pic_dir = "./face_person"
    cycleGAN = CycleGAN(para)
    IG_A = ImageGenerator(root=r'C:\PR_project\video_self\all',
                resize=para.resize, crop=para.crop)
    path_list = []
    for i in os.listdir(r'C:\PR_project\split_data_actor'):
        path_list.append(os.path.join(r'C:\PR_project\split_data_actor', i))
    image_generator_list = [ImageGenerator(root=i, resize=para.resize, crop=para.crop) for i in path_list]
    cycleGAN.fit(IG_A, image_generator_list)