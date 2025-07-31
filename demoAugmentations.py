from sklearn.model_selection import train_test_split

from Generators import *
from PlotMetrics import *
from DataPipeline import get_data_person, get_data_image, FaceImageAugmentor
from ModelHelpers import *


import numpy as np
import cv2

if __name__ == "__main__":
    # Example usage
    augmentor = None  # Replace with actual augmentor if needed
    min_pics = 10
    batch_size = 32
    epochs = 15
    triplet_count = 4000
    split = 0.1
    # Load dataset
    X_train, y_train, X_test, y_test, target_names = get_data_image(min_pics=min_pics, color=True, size=(160, 160), split=split)
    # X_train, y_train, X_test, y_test, target_names = get_data(min_pics=min_pics, test_size=0.20)

    augmentor_1 = FaceImageAugmentor(
        number_of_output=1,
        rotation_range=2, rotation_prob=0.08,  # very slight rotation, rare
        fliplr_prob=0.2,  # occasional horizontal flip
        brightness_range=(0.95, 1.05), brightness_prob=0.2,  # very slight brightness change
        y_translate_percentage=0.02, y_translate_prob=0.1,  # tiny vertical shift, rare
        x_translate_percentage=0.02, x_translate_prob=0.1,  # tiny horizontal shift, rare
        blur_percent=0.03, blur_prob=0.05,  # very slight blur, rare
        clahe_limit=1.0, clahe_prob=0.1,  # mild contrast enhancement, rare
        color=True
    )

    augmentor_2 = FaceImageAugmentor(
        number_of_output=1,
        rotation_range=3, rotation_prob=0.15,  # still subtle, a bit more frequent
        fliplr_prob=0.3,  # more flips
        brightness_range=(0.9, 1.1), brightness_prob=0.3,  # slightly wider brightness
        y_translate_percentage=0.03, y_translate_prob=0.15,
        x_translate_percentage=0.03, x_translate_prob=0.15,
        blur_percent=0.04, blur_prob=0.08,
        clahe_limit=1.2, clahe_prob=0.15,
        color=True
    )

    augmentor_3 = FaceImageAugmentor(
        number_of_output=1,
        rotation_range=5, rotation_prob=0.2,  # still mild, but most aggressive of the three
        fliplr_prob=0.4,
        brightness_range=(0.85, 1.15), brightness_prob=0.4,
        y_translate_percentage=0.04, y_translate_prob=0.2,
        x_translate_percentage=0.04, x_translate_prob=0.2,
        blur_percent=0.05, blur_prob=0.12,
        clahe_limit=1.5, clahe_prob=0.2,
        color=True
    )

    print('Augmenting data...')

    X_train_aug_1, y_train_aug_1 = augmentor_1.augment_batch(X_train, y_train)
    X_train_aug_2, y_train_aug_2 = augmentor_2.augment_batch(X_train, y_train)
    X_train_aug_3, y_train_aug_3 = augmentor_3.augment_batch(X_train, y_train)

    X_train = np.array(X_train, dtype=np.float32) / 255.0
    X_train_aug_1 = np.array(X_train_aug_1,dtype=np.float32)/255.0
    X_train_aug_2 = np.array(X_train_aug_2, dtype=np.float32) / 255.0
    X_train_aug_3 = np.array(X_train_aug_3, dtype=np.float32) / 255.0
    X_test = np.array(X_test, dtype=np.float32) / 255.0

    # if facenet uncomment -----------  !IMPORTANT! -------------------
    X_train = X_train * 2 - 1
    X_train_aug_1 = X_train_aug_1 *2 -1
    X_train_aug_2 = X_train_aug_2 * 2 - 1
    X_train_aug_3 = X_train_aug_3 * 2 - 1
    X_test = X_test * 2-1
    # endif facenet uncomment -----------------------------------------

    test_triplets = create_triplets(y_test, num_triplets=int(triplet_count * split))
    all_train_triplets = create_triplets(y_train_aug_1, num_triplets=triplet_count)
    train_triplets, val_triplets = train_test_split(all_train_triplets, test_size=0.1)
    #save_index = 55
    #img_bgr = cv2.cvtColor(X_train[save_index], cv2.COLOR_RGB2BGR)
    #cv2.imwrite("unaugmented_image.jpg", img_bgr)
    #img_bgr = cv2.cvtColor(X_train_aug_1[save_index], cv2.COLOR_RGB2BGR)
    #cv2.imwrite("augment_1_image.jpg", img_bgr)



    model_func = train_deep_triplet

    # Train model
    print("\nTraining model 1 with no augmentor")
    embedding_model_1, _ = model_func(X_train, train_triplets, val_triplets, batch_size=batch_size, epochs=epochs,plot=True)

    print("\nTraining model 2 with augmentor 1")
    embedding_model_2, _ = model_func(X_train_aug_1,train_triplets, val_triplets,batch_size=batch_size, epochs=epochs)

    print("\nTraining model 3 with augmentor 2")
    embedding_model_3, _ = model_func(X_train_aug_2,train_triplets, val_triplets,batch_size=batch_size, epochs=epochs)

    print("\nTraining model 4 with augmentor 3")
    embedding_model_4, _ = model_func(X_train_aug_3, train_triplets, val_triplets,batch_size=batch_size, epochs=epochs)

    #  Evaluate model
    # print("Performance on validation set")
    # threshold = evaluate_contrastive_model(embedding_model, val_triplets, X_train, y_train, target_names)
    #  Evaluate model
    # print("Performance on validation set")
    # threshold = evaluate_contrastive_model(embedding_model, val_triplets, X_train, y_train, target_names)

    print("\nModel 1: (no augment) Performance on test set")
    evaluate_model(embedding_model_1, test_triplets, X_test, y_test, target_names,plot=True)
    print("\nModel 2: (augmentor 1) Performance on test set")
    evaluate_model(embedding_model_2, test_triplets, X_test, y_test, target_names)
    print("\nModel 3: (augmentor 2) Performance on test set")
    evaluate_model(embedding_model_3, test_triplets, X_test, y_test, target_names)
    print("\nModel 4: (augmentor 3) Performance on test set")
    evaluate_model(embedding_model_4, test_triplets, X_test, y_test, target_names)