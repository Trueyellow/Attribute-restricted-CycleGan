# ISOBAR Deeplearning Group -- HKX
# This simple script is for clean our images and generating train and test data and save them into to file dict
import os
import shutil
import numpy as np

IMG_DIR = r"C:\PR_project\video_self_split_by_emotion"
IMG_SAVE_DIR = r"C:\PR_project\yh_face_emotion"
# train = 1-fraction
# test = fraction
SPLIT_FRACTION = 0.15


# This function is for cleaning the original image, it will delete GIF picture and
def generate_img_list(ori_img_dir):
    img_dir = os.listdir(ori_img_dir)
    print(img_dir)
    img_list = []
    for dir in img_dir:
        img_abs_path = ori_img_dir + "\\" + dir
        img_final_dir = os.listdir(img_abs_path)
        for imgs in img_final_dir:
            img_path = os.path.join(img_abs_path, imgs)
            img_list.append(img_path)
    return img_list


# Generate mask for split data
def random_choice(fraction, data):
    train_mask = np.random.choice(len(data), int(fraction * len(data)))
    test_data = (np.array(data)[train_mask]).tolist()
    train_data = list(set(data).difference(set(test_data)))
    return test_data, train_data


if __name__ == "__main__":
    line = generate_img_list(IMG_DIR)
    test_list, train_list = random_choice(SPLIT_FRACTION, line)
    print(" The length of original data: ", len(line), "\n",
          "The length of test data: ", len(test_list), "\n",
          "The length of train data: ", len(train_list), "\n",)

    # separate test and train, then save then into two file
    train_saved_dir = IMG_SAVE_DIR + "\\train\\"
    test_saved_dir = IMG_SAVE_DIR + "\\test\\"
    for train_path in train_list:
        path_split = train_path.split("\\")
        if not os.path.exists(train_saved_dir + path_split[-2]):
            os.makedirs(train_saved_dir + path_split[-2])
        shutil.copy(train_path, train_saved_dir + path_split[-2] + "\\" + path_split[-1])

    for test_path in test_list:
        path_split = test_path.split("\\")
        if not os.path.exists(test_saved_dir + path_split[-2]):
            os.makedirs(test_saved_dir + path_split[-2])
        shutil.copy(test_path, test_saved_dir + path_split[-2] + "\\" + path_split[-1])

