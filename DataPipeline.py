import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from collections import defaultdict
from tqdm import tqdm
import csv
from random import random
from random import uniform

face_cascade = None
coords_file_name = "face_crop_coords.csv"

class FaceImageAugmentor:
    '''
    This class augments the labeled faces in the wild dataset.
    Its main function takes and image, and returns a list of new images with augmentations applied randomly in
    accordance with the constructor parameters.
    Augmentations include:
    - translation
    - rotation
    - brightness
    - blur
    - clahe
    '''

    def __init__(
            self,
            number_of_output=10,
            rotation_range=15, rotation_prob=0.8,
            fliplr_prob=0.5,
            brightness_range=(0.7, 1.3), brightness_prob=0.7,
            y_translate_percentage=0.1, y_translate_prob=0.5,
            x_translate_percentage=0.1, x_translate_prob=0.5,
            blur_percent=0.03, blur_prob=0.3,
            clahe_limit=2.0, clahe_prob=0.4,
            color=True
    ):
        self.number_of_output = number_of_output
        self.rotation_range = rotation_range
        self.rotation_prob = rotation_prob
        self.fliplr_prob = fliplr_prob
        self.brightness_range = brightness_range
        self.brightness_prob = brightness_prob
        self.y_translate_percentage = y_translate_percentage
        self.y_translate_prob = y_translate_prob
        self.x_translate_percentage = x_translate_percentage
        self.x_translate_prob = x_translate_prob
        self.blur_percent = blur_percent
        self.blur_prob = blur_prob
        self.clahe_limit = clahe_limit
        self.clahe_prob = clahe_prob
        self.color = color

    def _to_grayscale(self, img):
        if hasattr(img, 'numpy'):
            img = img.numpy()

        img = np.array(img)

        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        if img.dtype != np.uint8:
            img = img.astype(np.uint8)

        return img

    def _apply_rotation(self, img):
        angle = uniform(-self.rotation_range, self.rotation_range)

        h_img, w_img = img.shape[:2]
        x = w_img//2
        y = h_img//2

        M = cv2.getRotationMatrix2D((x, y), angle, 1.0)
        rotated = cv2.warpAffine(
            img, M, (w_img, h_img),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        return cv2.resize(rotated, (h_img, w_img))

    def _apply_flip(self, img):
        return cv2.flip(img, 1)

    def _apply_brightness(self, img):
        factor = uniform(*self.brightness_range)
        img = img.astype(np.float32) * factor
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    def _apply_translation(self, img):
        tx = int(uniform(-self.x_translate_percentage, self.x_translate_percentage) * img.shape[1])
        ty = int(uniform(-self.y_translate_percentage, self.y_translate_percentage) * img.shape[0])
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        translated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return translated

    def _apply_blur(self, img):
        ksize = int(self.blur_percent * min(img.shape[:2]))
        ksize = max(ksize | 1, 3)
        return cv2.GaussianBlur(img, (ksize, ksize), 0)

    def _apply_clahe(self, img):
        clahe = cv2.createCLAHE(clipLimit=self.clahe_limit,tileGridSize=(8, 8))
        if img.ndim == 2:
            return clahe.apply(img)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        cl = clahe.apply(l)
        merged = cv2.merge((cl, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

    def augment(self, img):
        if not self.color:
            img = self._to_grayscale(img)
        augmented_images = []
        for _ in range(self.number_of_output):
            aug_img = img.copy()
            if random() < self.rotation_prob:
                aug_img = self._apply_rotation(aug_img)

            if random() < self.fliplr_prob:
                aug_img = self._apply_flip(aug_img)

            if random() < self.brightness_prob:
                aug_img = self._apply_brightness(aug_img)

            if random() < self.x_translate_prob or random() < self.y_translate_prob:
                aug_img = self._apply_translation(aug_img)

            if random() < self.blur_prob:
                aug_img = self._apply_blur(aug_img)

            if random() < self.clahe_prob:
                aug_img = self._apply_clahe(aug_img)

            augmented_images.append(aug_img)

        return augmented_images

    def augment_batch(self, X, y):
        N= len(X)
        shapeX = list(X.shape)
        shapeX[0] = N*self.number_of_output
        shapeX = tuple(shapeX)
        shapeY = list(y.shape)
        shapeY[0] = N * self.number_of_output
        shapeY = tuple(shapeY)
        augmented_images = np.empty(shapeX,dtype=np.float32)
        augmented_labels = np.empty(shapeY)
        for i in range(N):
            for j,img in enumerate(self.augment(X[i])):
                augmented_images[i+j*N] = img
                augmented_labels[i+j*N] = y[i]
        return augmented_images, augmented_labels




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


def crop_default(img, c, w, size):
    """
    Crop in case of no augmentor
    :param img: np Image
    :param c: centroid shape (x,y)
    :param w: width
    :param size: resulting img size
    :return: np Image
    """
    img_h, img_w = img.shape[:2]
    x1 = int(round(c[0] - w / 2))
    y1 = int(round(c[1] - w / 2))
    x2 = int(round(c[0] + w / 2))
    y2 = int(round(c[1] + w / 2))
    # Ensure crop is within image bounds
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, img_w), min(y2, img_h)
    cropped = img[y1:y2, x1:x2]
    return cv2.resize(cropped,size)

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

def get_data_person(min_pics = 20, color= False, size=(64,64), split=0.2):
    """
    Splits data set by person
    :param augmentor: FaceImageAugmentor object
    :param min_pics: minimum number of pics per person to return their pictures
    :param color: fall back if no augmentor, output color images or grayscale
    :param size: fall back if no augmentor, output image size
    :param sample: show debug plot
    :return: np Array of Images shape (n,h,w,c), labels reference to a person's #, list of names bases on person's #
    """
    ds, ds_info = tfds.load("lfw", split='train', as_supervised=True, with_info=True)
    results = []
    # Face cropping is an expensive task so we only run it once and save the output to a csv
    if not os.path.exists(coords_file_name):
        counter = 0
        failed = 0
        print("Calculating Faces Centers")
        # Run face centering code first and save it to a csv
        global face_cascade
        face_cascade= cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
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

    # All centroid, with pairs should be loaded now
    res_dict={}
    for row in results:
        res_dict[row[0]] = (row[1], (row[2], row[3]), row[4])

    person_dict_num = defaultdict(int) # Unique non augmented pics per person
    person_dict_images = defaultdict(list)
    valid_names =[]
    valid_names_dict={}
    i=0
    for lbl,img in tqdm(ds,total=len(ds)):
        if i in res_dict.keys():
            (name, c, w) = res_dict[i]
            person_dict_num[name] += 1
            person_dict_images[name].append(crop_default(img.numpy(),c,w,size))
        i+=1
    num_people_test=0
    for name in person_dict_num.keys():
        if person_dict_num[name]>=min_pics:
            valid_names_dict[name]=len(valid_names)
            valid_names.append(name)
            person_dict_num[name] = 0
            if np.random.random() < split:
                person_dict_num[name] = 1
                num_people_test += 1


    train_i=[]
    train_l=[]
    test_i=[]
    test_l=[]
    for name in valid_names:
        for img in person_dict_images[name]:
            if person_dict_num[name]:
                test_i.append(img)
                test_l.append(valid_names_dict[name])
            else:
                train_i.append(img)
                train_l.append(valid_names_dict[name])
    print(f"Number people in Test Set: {num_people_test}")
    del ds
    return np.array(train_i), np.array(train_l) , np.array(test_i), np.array(test_l), valid_names

def get_data_image(min_pics = 20, color= False, size=(64,64), split=0.2):
    """
    Splits data set by image
    :param min_pics: minimum number of pics per person to return their pictures
    :param color: output color images or grayscale
    :param size: output image size
    :param split: chance for train/test split
    :return: np Array of Images shape (n,h,w,c), labels reference to a person's #, list of names bases on person's #
    """
    ds, ds_info = tfds.load("lfw", split='train', as_supervised=True, with_info=True)
    results = []
    # Face cropping is an expensive task so we only run it once and save the output to a csv
    if not os.path.exists(coords_file_name):
        counter = 0
        failed = 0
        print("Calculating Faces Centers")
        # Run face centering code first and save it to a csv
        global face_cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
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

    # All centroid, with pairs should be loaded now
    res_dict={}
    for row in results:
        res_dict[row[0]] = (row[1], (row[2], row[3]), row[4])

    person_dict_num = defaultdict(int) # Unique non augmented pics per person
    person_dict_images = defaultdict(list)
    valid_total_images = 0
    valid_names =[]
    valid_names_dict={}
    i=0
    for lbl,img in tqdm(ds,total=len(ds)):
        if i in res_dict.keys():
            (name, c, w) = res_dict[i]
            person_dict_num[name] += 1

            person_dict_images[name].append(crop_default(img.numpy(),c,w,size))
        i+=1

    for name in person_dict_num.keys():
        if person_dict_num[name] >= min_pics:
            valid_names_dict[name] = len(valid_names)
            valid_names.append(name)
            valid_total_images +=  person_dict_num[name]

    samples_done =0
    usable_names = valid_names.copy()
    train_i = []
    train_l = []
    test_i = []
    test_l = []
    while samples_done < valid_total_images:
        name = np.random.choice(usable_names)
        dest = np.random.random() < split # False Train
        if person_dict_num[name]==3 or person_dict_num[name] == 2:
            if dest:
                test_i.extend(person_dict_images[name])
                test_l.extend([valid_names_dict[name]]*person_dict_num[name])
            else:
                train_i.extend(person_dict_images[name])
                train_l.extend([valid_names_dict[name]] * person_dict_num[name])
            samples_done+=person_dict_num[name]
            person_dict_num[name] = 0
            person_dict_images[name] = []
            usable_names.remove(name)
        else:
            c, b = np.random.choice(range(person_dict_num[name]), 2, replace=False)
            a = max(c,b)
            b = min(c,b)
            if dest:
                test_i.append(person_dict_images[name].pop(a))
                test_i.append(person_dict_images[name].pop(b))
                test_l.extend([valid_names_dict[name]] * 2)
            else:
                train_i.append(person_dict_images[name].pop(a))
                train_i.append(person_dict_images[name].pop(b))
                train_l.extend([valid_names_dict[name]] * 2)
            person_dict_num[name] -= 2
            samples_done +=2
    del ds
    return np.array(train_i), np.array(train_l) , np.array(test_i), np.array(test_l), valid_names


if __name__ == '__main__':
    augmentor = FaceImageAugmentor(
        number_of_output=1,
        rotation_range=8, rotation_prob=0.2,
        fliplr_prob=0.5,
        brightness_range=(0.7, 1.3), brightness_prob=0.7,
        y_translate_percentage=0.1, y_translate_prob=0.2,
        x_translate_percentage=0.1, x_translate_prob=0.2,
        blur_percent=0.08, blur_prob=0.4,
        clahe_limit=0.8, clahe_prob=0.4
    )
    X_train, y_train, X_test, y_test, names = get_data_image(min_pics=2)
    X_test, y_test = augmentor.augment_batch(X_test,y_test)
    N = len(X_test)
    num_images = min(30, N)
    idxs = np.random.choice(N, num_images, replace=False)
    rows = int(np.ceil(num_images / 5))
    fig, axes = plt.subplots(rows, 5, figsize=(5 * 2, rows * 2))
    axes = axes.flatten()
    for ax, idx in zip(axes, idxs):
        img = X_test[idx]
        lab = y_test[idx]
        if img.ndim == 2:
            ax.imshow(img, cmap='gray')
        else:
            ax.imshow(img)
        ax.set_title(names[lab], fontsize=12)
        ax.axis('off')

    # turn off any unused subplots
    for ax in axes[num_images:]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()


