# Pattern Recognition final project Group 8 Kaixiang Huang
# For generating DEMO images

import cv2
from build_DG import resnet_6blocks_A
import numpy as np
from util import vis_grid, label_generate
import os
from parameter_handle import parameter
from util import ImageGenerator

para = parameter()
weight = "./saved_weight/b2a_yh.h5"
para.resize = (128, 128)

label = 3

generate_position = r'C:\PR_project\generate_yh\generate_yh_no_cnn\{}.png'
path1 = r'C:\PR_project\generate_yh\generate_yh_no_cnn'
test_image_root = r"C:\PR_project\angry_woman"

image_b = ImageGenerator(root=test_image_root, resize=para.resize, crop=None)
gen_A = resnet_6blocks_A(para.shapeB, para.label_shape_G, para.shapeB[2], ngf=para.ngf, name='gen_A')
gen_A.load_weights(weight)
img_b_generator = image_b(para.batch_size, random=False)
count = 0
for i in img_b_generator:
    label_G_true, _, _ = label_generate(label, para.label_num, para.label_shape_G, para.label_shape_D)
    imgb2a = gen_A.predict([i[np.newaxis, ...], label_G_true])
    vis_grid(np.concatenate([imgb2a, i[np.newaxis, ...]], axis=0), 2, 1,
             generate_position.format(count, label))
    count += 1


frame_list = os.listdir(path1)
frame_list.sort(key=lambda x: int(x[:-4]))
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter(r"C:\PR_project\generate_yh\demo_video\yh_without_cnn_sad.avi", fourcc, 30.0, (128, 256))
for i in frame_list:
    frame_tmp = cv2.imread(os.path.join(path1, i))
    out.write(frame_tmp)
out.release()