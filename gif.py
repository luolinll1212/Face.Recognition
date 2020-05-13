# *_*coding:utf-8 *_*
import os
import imageio

def main():

    frames = []

    for i in range(1, 101):
        image_name = "./images/epoch=" + str(i) + ".jpg"
        # print(img_name)
        frames.append(imageio.imread(image_name))
    imageio.mimsave('centerloss_3.gif', frames, 'GIF', duration=0.35)


if __name__ == '__main__':
    main()