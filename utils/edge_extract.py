import cv2
import os
import numpy as np

def get_edge(imgs_list, imgs_path, edge_path):
    save_path = edge_path
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    if not os.path.exists(imgs_path):
        raise FileNotFoundError(f'Dir {imgs_path} not found!')
    
    num = len(imgs_list)
    for i, img_name in enumerate(imgs_list):
        
        img_path = os.path.join(imgs_path, img_name)
        img =cv2.imread(img_path, 0)
        img[img>127.5] = 255
        img[img<=127.5] = 0
    

        kernel = np.ones((5, 5), np.uint8)  
        eroded = cv2.erode(img, kernel, iterations=1)

        sharp_kernel = np.array([[0, -1, 0],
                                [-1, 5, -1],
                                [0, -1, 0]], np.float32) 
        sharpened = cv2.filter2D(img, -1, sharp_kernel)

        edge = eroded - sharpened
        edge[edge>0] = 255
    
        edge_save_path = os.path.join(save_path, img_name)
        cv2.imwrite(edge_save_path, edge)
        print(f'{i}----->{num}')

def get_img_list(imgs_path):
    imgs_name = sorted(os.listdir(imgs_path))
    return imgs_name


if __name__ == '__main__':
    image_path = ''
    edge_path = ''
    list = get_img_list(imgs_path=image_path)
    get_edge(list, image_path, edge_path)
