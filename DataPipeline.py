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

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
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
            output_size=(64, 64),
            number_of_output=10,
            rotation_range=15, rotation_prob=0.8,
            fliplr_prob=0.5,
            brightness_range=(0.7, 1.3), brightness_prob=0.7,
            y_translate_percentage=0.1, y_translate_prob=0.5,
            x_translate_percentage=0.1, x_translate_prob=0.5,
            blur_percent=0.03, blur_prob=0.3,
            clahe_limit=2.0, clahe_prob=0.4,
            color=False
    ):
        self.output_size = output_size
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
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if img.dtype != np.uint8:
            img = img.astype(np.uint8)

        return img

    def _crop_and_resize(self, img, c, w, rot=0):
        img = self._to_grayscale(img)
        w = int(w)
        h_img, w_img = img.shape[:2]
        x, y = c
        M = cv2.getRotationMatrix2D((x, y), rot, 1.0)
        rotated = cv2.warpAffine(img, M, (w_img, h_img), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

        x1 = int(round(x - w / 2))
        y1 = int(round(y - w / 2))
        x2 = int(round(x + w / 2))
        y2 = int(round(y + w / 2))

        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, w_img), min(y2, h_img)

        cropped = rotated[y1:y2, x1:x2]

        if cropped.size == 0 or cropped.shape[0] == 0 or cropped.shape[1] == 0:
            y_start = max(int(round(y - w / 2)), 0)
            y_end = min(int(round(y + w / 2)), h_img)
            x_start = max(int(round(x - w / 2)), 0)
            x_end = min(int(round(x + w / 2)), w_img)
            cropped = rotated[y_start:y_end, x_start:x_end]

            if cropped.size == 0 or cropped.shape[0] == 0 or cropped.shape[1] == 0:
                cropped = rotated

        resized = cv2.resize(cropped, self.output_size)
        return resized

    def _apply_rotation(self, img, center, width):
        img = self._to_grayscale(img)
        angle = uniform(-self.rotation_range, self.rotation_range)

        h_img, w_img = img.shape[:2]
        x, y = center

        M = cv2.getRotationMatrix2D((x, y), angle, 1.0)
        rotated = cv2.warpAffine(
            img, M, (w_img, h_img),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        return cv2.resize(rotated, self.output_size)

    def _apply_flip(self, img):
        img = self._to_grayscale(img)
        return cv2.flip(img, 1)

    def _apply_brightness(self, img):
        img = self._to_grayscale(img)
        factor = uniform(*self.brightness_range)
        img = img.astype(np.float32) * factor
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    def _apply_translation(self, img):
        img = self._to_grayscale(img)
        tx = int(uniform(-self.x_translate_percentage, self.x_translate_percentage) * img.shape[1])
        ty = int(uniform(-self.y_translate_percentage, self.y_translate_percentage) * img.shape[0])
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        translated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return translated

    def _apply_blur(self, img):
        img = self._to_grayscale(img)
        ksize = int(self.blur_percent * min(img.shape[:2]))
        ksize = max(ksize | 1, 3)
        return cv2.GaussianBlur(img, (ksize, ksize), 0)

    def _apply_clahe(self, img):
        img = self._to_grayscale(img)
        clahe = cv2.createCLAHE(clipLimit=self.clahe_limit, tileGridSize=(8, 8))
        return clahe.apply(img)

    def augment(self, img, center, width):
        img = img.numpy()
        if not self.color:
            img = self._to_grayscale(img)
        augmented_images = []
        for _ in range(self.number_of_output):
            aug_img = self._crop_and_resize(img, center, width)

            if random() < self.rotation_prob:
                aug_img = self._apply_rotation(aug_img, center, width)

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

def get_data(augmentor = None, min_pics = 20, color= False, size=(64,64), sample=False):
    """
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
    images = []
    labels = []
    names_dict = {}
    names = []
    for i in range(len(ds)):
        if i in res_dict.keys():
            (name, _, _) = res_dict[i]
            person_dict_num[name] += 1

    i=0
    for _,img in tqdm(ds, total=len(ds)):
        if i in res_dict.keys():
            (name, c, w) = res_dict[i]
            if person_dict_num[name] >= min_pics:
                if name not in names_dict.keys():
                    names_dict[name] = len(names)
                    names.append(name)
                img_list = []
                if augmentor is not None:
                    img_list = augmentor.augment(img, center=c, width=w)
                else:
                    img_np = img.numpy()
                    if not color:
                        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                    img_list.append(crop_default(img_np, c, w, size))
                images.extend(img_list)
                labels.extend([names_dict[name]]*len(img_list))
        i+=1

    if sample:

        randList = np.random.choice(names, size=(min(30,len(names))), replace=False, p=None)
        randImgs = []
        pic_dict = defaultdict(list)
        for i, label in enumerate(labels):
            pic_dict[names[label]].append(i)
        for person in randList:
            temp= np.random.randint(0,len(pic_dict[person]))
            randImgs.append(images[pic_dict[person][temp]])
        plot_all_person_images(randImgs,"Rand")
    out_imgs = np.array(images,dtype =np.float16)/255.0
    if len(out_imgs.shape) == 3: # If gray scale function deleted channel instead of leaving it as 1
        out_imgs = np.expand_dims(out_imgs, -1)  # (n_samples, h, w, c)
    return out_imgs, np.array(labels), names

if __name__ == '__main__':
    augmentor = FaceImageAugmentor(
        number_of_output=5,
        rotation_range=8, rotation_prob=0.5,
        fliplr_prob=0.5,
        brightness_range=(0.7, 1.3), brightness_prob=0.7,
        y_translate_percentage=0.1, y_translate_prob=0.2,
        x_translate_percentage=0.1, x_translate_prob=0.2,
        blur_percent=0.08, blur_prob=0.4,
        clahe_limit=0.8, clahe_prob=0.4
    )
    get_data(augmentor,min_pics=2,color=False,sample=True)
    pass

