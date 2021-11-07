# Name: Sai Anish Garapati
# UIN: 650208577

import torch, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as image


def split_images_to_training_and_test(dir_path):
    img_list = os.listdir(dir_path)
    print(len(img_list))



if __name__ == '__main__':
    split_images_to_training_and_test('./output/')
