import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from collections import defaultdict
from tqdm import tqdm
import csv


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


coords_file_name = "face_crop_coords.csv"

augment_rots = [0.0]
augment_brightness = [1.0]

def crop_face_opencv(img_np):
    """
    very rudimentary face cropping with haar cascades
    :param img_np: np Image
    :return: centroid shape (x,y) , width
    """
    if img_np.ndim == 2:
        gray = img_np
    else:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    # Fast to Slow this catches most faces
    faces = face_cascade.detectMultiScale(gray, 1.05, 7,minSize=(64, 64))
    if len(faces) == 0:
        faces = face_cascade.detectMultiScale(gray, 1.01, 7, minSize=(64, 64))
        if len(faces) == 0:
            gray = cv2.equalizeHist(gray)
            faces = face_cascade.detectMultiScale(gray, 1.01, 7, minSize=(64, 64))
    face = (0,0,0,0)
    # Get the largest face, it's probably the subject
    for (x, y, w, h) in faces:
        if w> face[2]:
            face = (x, y, w, h)
    c = (face[0]+face[2]/2,face[1]+face[3]/2)
    if face[2]!=face[3] or face[2] == 0:
        return None, None
    return c, face[2] * 1.2


def crop(img, c, w, rot,size):
    """
    Crop image with augments
    :param img: np Image
    :param c: centroid shape (x,y)
    :param w: width
    :param rot: rotation
    :param size: resulting img size
    :return: np Image
    """
    w = int(w)
    h_img, w_img = img.shape[:2]
    x, y = c
    M = cv2.getRotationMatrix2D((x, y), rot, 1.0)
    rotated = cv2.warpAffine(img, M, (w_img, h_img), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    x1 = int(round(x - w / 2))
    y1 = int(round(y - w / 2))
    x2 = int(round(x + w / 2))
    y2 = int(round(y + w / 2))

    #Ensure crop is within image bounds
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, w_img), min(y2, h_img)

    #Crop
    cropped = rotated[y1:y2, x1:x2]
    return cv2.resize(cropped,size)

def img_augmentation(img, c, w, size,rots,brts,flip):
    """
    Got through list of augments and augment image
    :param img: np Image
    :param c: centroid
    :param w: width
    :param size: output image size
    :param rots: List of rotations to make
    :param brts: List of brightness alterations to make
    :param flip: boolean to flip image too
    :return: List of augmented images
    """
    ret_list=[]
    for rot in rots:
        rot_img = crop(img, c, w, rot, size)
        for brt in brts:
            brt_img = rot_img.astype(np.float32)
            brt_img = brt_img * brt
            brt_img = np.clip(brt_img, 0, 255)
            brt_img.astype(np.uint8)
            ret_list.append(brt_img)
            if flip:
                flipped_img = cv2.flip(brt_img, 1)
                ret_list.append(flipped_img)
    return ret_list

def plot_all_person_images(img_list, label, max_cols=10, figsize=(15, 3)):
        #plot images for debug
        if len(img_list) == 0:
            return
        n_imgs = len(img_list)
        n_cols = min(max_cols, n_imgs)
        n_rows = (n_imgs + n_cols - 1) // n_cols
        plt.figure(figsize=(figsize[0], figsize[1]*n_rows))
        for i, img in enumerate(img_list):
            plt.subplot(n_rows, n_cols, i+1)
            if img.ndim == 2:
                plt.imshow(img, cmap='gray')
            else:
                plt.imshow(img)
            plt.axis('off')
        plt.suptitle(f"{label} ({n_imgs} images)", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.show()

def get_data(min_pics = 20, color= False, size=(64,64), rots=[0.0],brts=[1.0],sample=False, flip = False):
    """

    :param min_pics: minimum number of pics per person to return their pictures
    :param color: output color images or grayscale
    :param size: output image size
    :param rots: list of rotations augments to make
    :param brts: list of brightness alterations to make
    :param sample: show debug plot
    :param flip: add flipped images
    :return: np Array of Images shape (n,h,w,c), labels reference to a person's #, list of names bases on person's #
    """
    ds, ds_info = tfds.load("lfw", split='train', as_supervised=True, with_info=True)
    results = []
    if not os.path.exists(coords_file_name):
        counter = 0
        failed = 0
        print("Calculating Faces Centers")
        # Run face centering code first and save it to a csv
        for label, img, in tqdm(ds, total=len(ds)):
            img_np = img.numpy()
            if not color:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            name = label.numpy().decode("utf-8")
            c, w = crop_face_opencv(img_np)
            if c is not None:
                results.append([counter, name, c[0], c[1], w])
            else:
                print("failed")
                failed += 1
            counter += 1
        print(f"Failed {failed} out of {len(ds)}")
        with open(coords_file_name, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['index', 'name', 'c_x', 'c_y', 'w'])
            writer.writerows(results)
    else:
        # read csv
        with open(coords_file_name, 'r') as file:
            my_reader = csv.reader(file, delimiter=',')
            next(my_reader)
            for row in my_reader:
                (index, name, cx, cy, w) = row
                index = int(index)
                cx = float(cx)
                cy = float(cy)
                w = float(w)
                results.append([index, name, cx, cy, w])
    res_dict={}
    for row in results:
        res_dict[row[0]] = (row[1], (row[2], row[3]), row[4])
    person_dict_img = defaultdict(list)
    person_dict_num = defaultdict(int)
    counter = 0
    # Do augmentations
    print("Augmenting Data")
    for _, img, in tqdm(ds, total=len(ds)):
        if counter in res_dict.keys():
            img_np = img.numpy()
            if not color:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            (name, c, w) = res_dict[counter]
            img_list = img_augmentation(img_np, c, w, size,rots,brts,flip)
            person_dict_num[name] += 1
            person_dict_img[name].extend(img_list)
        counter += 1
    images = []
    labels = []
    names = []
    counter = 0
    for key in person_dict_img.keys():
        if person_dict_num[key] >= min_pics:
            names.append(key)
            images.extend(person_dict_img[key])
            labels.extend([counter]*len(person_dict_img[key]))
            counter += 1
    if sample:
        randList = np.random.choice(list(person_dict_num.keys()), size=(100), replace=True, p=None)
        randImgs = []
        for person in randList:
            temp= np.random.randint(0,len(person_dict_img[person]))
            randImgs.append(person_dict_img[person][temp])
        plot_all_person_images(randImgs,"Rand")
    out_imgs = np.array(images)/255.0
    if len(out_imgs.shape) == 3:
        out_imgs = np.expand_dims(out_imgs, -1)  # (n_samples, h, w, c)
    return out_imgs, np.array(labels), names

if __name__ == '__main__':
    get_data(min_pics=40,color=False,rots=[-45,0],brts=[0.7,1.0],sample=True)
    pass