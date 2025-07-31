from numpy.f2py.crackfortran import verbose
from sklearn.model_selection import train_test_split
import gc
from Generators import *
from PlotMetrics import *
from DataPipeline import get_data_person, get_data_image, FaceImageAugmentor
from ModelHelpers import *

import tensorflow as tf

import numpy as np
import cv2

if __name__ == "__main__":
    # Example usage
    min_pics = 2
    batch_size = 128
    epochs = 30
    triplet_count = 300000
    split = 0.1
    # Load dataset
    X_train, y_train, X_test, y_test, target_names = get_data_image(min_pics=min_pics, color=True, size=(160, 160), split=split)
    gc.collect()
    augmentor = FaceImageAugmentor(
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
    X_train, y_train = augmentor.augment_batch(X_train, y_train)

    X_train /= 255.0
    X_test = X_test.astype(np.float32) / 255.0

    # if facenet uncomment -----------  !IMPORTANT! -------------------
    X_train *=2
    X_train -=1
    X_test *= 2
    X_test -= 1
    # endif facenet uncomment -----------------------------------------

    test_tri = create_triplets(y_test, num_triplets=int(triplet_count * split))
    all_train_tri = create_triplets(y_train, num_triplets=triplet_count)
    train_tri, val_tri = train_test_split(all_train_tri, test_size=0.1)

    #save_index = 55
    #img_bgr = cv2.cvtColor(X_train[save_index], cv2.COLOR_RGB2BGR)
    #cv2.imwrite("unaugmented_image.jpg", img_bgr)
    #img_bgr = cv2.cvtColor(X_train_aug_1[save_index], cv2.COLOR_RGB2BGR)
    #cv2.imwrite("augment_1_image.jpg", img_bgr)

    runs =[
        #{"mf":train_simple_contrastive,"pn":"Simple Contrastive"},
        #{"mf":train_deep_contrastive,"pn":"Deep Contrastive"},
        #{"mf": train_simple_triplet, "pn": "Simple Triplet"},
        {"mf": train_deep_triplet, "pn": "Deep Triplet"},
        #{"mf": train_facenet, "pn": "FaceNet PreTrain"},
    ]

    summary =[]
    for i, run in enumerate(runs):
        print(f"Training model {i}: " + run["pn"])
        embedding_model_1, _ = run["mf"](X_train, train_tri, val_tri, batch_size=batch_size, epochs=epochs,plot=run["pn"])
        res = evaluate_model(embedding_model_1, test_tri, X_test, y_test, target_names,verbose=False, plot=True)
        res["pn"] =  run["pn"]
        summary.append(res)

    model_func = train_deep_triplet

    for run in summary:
        name = run["pn"]
        acc= run["accuracy"]
        prec= run["precision"]
        rec= run["recall"]
        f1= run["f1"]
        roc_auc= run["roc_auc"]
        print(f"Name: {name}, Accuracy: {acc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}, ROC AUC: {roc_auc:.3f}")