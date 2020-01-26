import numpy as np
import pandas as pd

from pathlib import Path

from itertools import groupby
import cv2
import os
import torchvision

category_num = 46 + 1

class ImageMask():
    masks = {}
    
    def make_mask_img(self, segment_df):
        seg_width = segment_df.iloc[0].Width
        seg_height = segment_df.iloc[0].Height
        
        seg_img = np.copy(self.masks.get((seg_width, seg_height)))
        try:
            if not seg_img:
                seg_img = np.full(seg_width*seg_height, category_num-1, dtype=np.int32)
                self.masks[(seg_width, seg_height)] = np.copy(seg_img)
        except:
            # seg_img exists
            pass
        for encoded_pixels, class_id in zip(segment_df["EncodedPixels"].values, segment_df["ClassId"].values):
            pixel_list = list(map(int, encoded_pixels.split(" ")))
            for i in range(0, len(pixel_list), 2):
                start_index = pixel_list[i] - 1
                index_len = pixel_list[i+1] - 1
                if int(class_id.split("_")[0]) < category_num - 1:
                    seg_img[start_index:start_index+index_len] = int(class_id.split("_")[0])
        seg_img = seg_img.reshape((seg_height, seg_width), order='F')
        return seg_img
        

def create_label(images, path_lbl, path_img):
    """
    img_name = "53d0ee82b3b7200b3cec8c3c1becead9.jpg"
    img_df = df[df.ImageId == img_name]
    img_mask = ImageMask()
    mask = img_mask.make_mask_img(img_df)
    """
    img_mask = ImageMask()

    print("Start creating label")
    for i,img in enumerate(images):
        fname = path_lbl.joinpath(os.path.splitext(img)[0] + '_P.png').as_posix() 
        if os.path.isfile(fname):
            continue
        test_f = path_img.joinpath(os.path.splitext(img)[0] + '.jpg').as_posix()
        if not os.path.isfile(test_f):
            continue
        img_df = df[df.ImageId == img]
        mask = img_mask.make_mask_img(img_df)
        img_mask_3_chn = np.dstack((mask, mask, mask))
        cv2.imwrite(fname, img_mask_3_chn)
        if i % 40 ==0 : print(i, end=" ")
    print("Finish creating label")
            
        
def get_predictions(path_test, learn, size):
    # predicts = get_predictions(path_test, learn)
    learn.model.cuda()
    files = list(path_test.glob("**/*.jpg"))    
    test_count = len(files)
    results = {}
    for i, img in enumerate(files):
        results[img.stem] = learn.predict(open_image(img).resize(size))[1].data.numpy().flatten()
    
        if i%20==0:
            print("\r{}/{}".format(i, test_count), end="")
    return results       
        
def encode(input_string):
    return [(len(list(g)), k) for k,g in groupby(input_string)]

def run_length(label_vec):
    encode_list = encode(label_vec)
    index = 1
    class_dict = {}
    for i in encode_list:
        if i[1] != category_num-1:
            if i[1] not in class_dict.keys():
                class_dict[i[1]] = []
            class_dict[i[1]] = class_dict[i[1]] + [index, i[0]]
        index += i[0]
    return class_dict

    
def get_submission_df(predicts):
    sub_list = []
    for img_name, mask_prob in predicts.items():
        class_dict = run_length(mask_prob)
        if len(class_dict) == 0:
            sub_list.append([img_name, "1 1", 1])
        else:
            for key, val in class_dict.items():
                sub_list.append([img_name + ".jpg", " ".join(map(str, val)), key])
        # sub_list
    jdf = pd.DataFrame(sub_list, columns=['ImageId','EncodedPixels', 'ClassId'])
    return jdf
        
        
def test_mask_to_img(segment_df):
    """
    plt.imshow(test_mask_to_img(jdf))
    """
    seg_img = np.full(size*size, category_num-1, dtype=np.int32)
    for encoded_pixels, class_id in zip(segment_df["EncodedPixels"].values, segment_df["ClassId"].values):
        encoded_pixels= str(encoded_pixels)
        class_id = str(class_id)
        
        pixel_list = list(map(int, encoded_pixels.split(" ")))
        for i in range(0, len(pixel_list), 2):
            start_index = pixel_list[i] - 1
            index_len = pixel_list[i+1] - 1
            if int(class_id.split("_")[0]) < category_num - 1:
                seg_img[start_index:start_index+index_len] = int(class_id.split("_")[0])
        seg_img = seg_img.reshape((size, size), order='F')
    return seg_img
    
