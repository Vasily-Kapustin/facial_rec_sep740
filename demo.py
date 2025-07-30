import cv2
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from sklearn.model_selection import train_test_split
import tensorflow.keras.backend as K

from Generators import *
from PlotMetrics import *
from DataPipeline import get_data, get_data_person, FaceImageAugmentor
from ContrastiveLoss import contrastive_loss

def train_contrastive_model(X, y, train_triplets, test_triplets, batch_size=64, epochs=10):
    """
    Builds and trains the contrastive (Siamese) model.
    Returns the trained embedding model and the contrastive network.
    """
    # Create embedding model
    embedding_model = tf.keras.Sequential([
        layers.Input(shape=X.shape[1:]),
        layers.Conv2D(32, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dense(64),
        layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)),
    ], name="EmbeddingModel")

    embedding_model.summary()

    # Build Siamese network
    input_a = Input(shape=X.shape[1:])
    input_b = Input(shape=X.shape[1:])

    embedding_a = embedding_model(input_a)
    embedding_b = embedding_model(input_b)

    merged_output = layers.Lambda(lambda x: tf.sqrt(tf.reduce_sum(tf.square(x[0] - x[1]), axis=1, keepdims=True)))(
        [embedding_a, embedding_b]
    )
    contrast_net = Model([input_a, input_b], merged_output)
    contrast_net.compile(loss=contrastive_loss, optimizer='adam')

    # Train/test split for triplets
    
    train_gen = pair_generator(X, train_triplets, batch_size)
    test_gen = pair_generator(X, test_triplets, batch_size)
    steps_per_epoch = len(train_triplets) // batch_size
    validation_steps = len(test_triplets) // batch_size

    # Train model
    print("Training model")
    H2 = contrast_net.fit(
        train_gen,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=test_gen,
        validation_steps=validation_steps
    )
    plot_training_history(H2, "Contrastive")

    return embedding_model, contrast_net, train_triplets, test_triplets

def evaluate_contrastive_model(embedding_model, test_triplets, X, y, target_names):
    """
    Evaluates the trained embedding model and plots results.
    """
    print("Evaluating model")
    pairs, pair_labels = pairs_from_triplets(test_triplets)
    threshold = evaluate_verification(embedding_model, pairs, pair_labels, X)
    pairs, pair_labels = pairs_from_triplets(test_triplets, n_pairs=8, other=True)
    plot_face_pairs(embedding_model, pairs, pair_labels, X, y, target_names, threshold=0.5)
    return threshold

def run_contrastive_experiment(augmentor, min_pics=2, batch_size=64, epochs=10):
    """
    Runs the contrastive loss experiment with configurable augmentor, min_pics, and batch_size.
    Returns the trained embedding model and evaluation results.
    """
    # Load dataset
    X, y, target_names = get_data(augmentor, min_pics=min_pics)

    # Create triplets (anchor, positive, negative)
    print("Generating triplets")
    triplets = create_triplets(y, num_triplets=15000)
    print(f"Generated {len(triplets)} triplets")

    # Train model
    embedding_model, contrast_net, test_triplets, triplets = train_contrastive_model(
        X, y, triplets, batch_size=batch_size, epochs=epochs
    )

    # Evaluate model
    threshold = evaluate_contrastive_model(embedding_model, test_triplets, triplets, X, y, target_names)

    return embedding_model, threshold

if __name__ == "__main__":
    # Example usage
    augmentor = None  # Replace with actual augmentor if needed
    min_pics = 2
    batch_size = 64
    epochs = 2

    # Load dataset
    #X, y, target_names = get_data(augmentor, min_pics=min_pics, color=True)#, test_size=0.20)
    #X, y, target_names = get_data(augmentor,min_pics=2)
    X_train, y_train, X_test, y_test, target_names = get_data_person(min_pics = 20, color = True, size=(64,64), split=0.2)

    #X_train, y_train, X_test, y_test, target_names = get_data(min_pics=min_pics, test_size=0.20)

    augmentor_1 = FaceImageAugmentor(
        number_of_output=1,
        rotation_range=2, rotation_prob=0.08,    # very slight rotation, rare
        fliplr_prob=0.2,                         # occasional horizontal flip
        brightness_range=(0.95, 1.05), brightness_prob=0.2,  # very slight brightness change
        y_translate_percentage=0.02, y_translate_prob=0.1,   # tiny vertical shift, rare
        x_translate_percentage=0.02, x_translate_prob=0.1,   # tiny horizontal shift, rare
        blur_percent=0.03, blur_prob=0.05,       # very slight blur, rare
        clahe_limit=1.0, clahe_prob=0.1,         # mild contrast enhancement, rare
        color=True
    )

    augmentor_2 = FaceImageAugmentor(
        number_of_output=1,
        rotation_range=3, rotation_prob=0.15,    # still subtle, a bit more frequent
        fliplr_prob=0.3,                         # more flips
        brightness_range=(0.9, 1.1), brightness_prob=0.3,    # slightly wider brightness
        y_translate_percentage=0.03, y_translate_prob=0.15,
        x_translate_percentage=0.03, x_translate_prob=0.15,
        blur_percent=0.04, blur_prob=0.08,
        clahe_limit=1.2, clahe_prob=0.15,
        color=True
    )

    augmentor_3 = FaceImageAugmentor(
        number_of_output=1,
        rotation_range=5, rotation_prob=0.2,     # still mild, but most aggressive of the three
        fliplr_prob=0.4,
        brightness_range=(0.85, 1.15), brightness_prob=0.4,
        y_translate_percentage=0.04, y_translate_prob=0.2,
        x_translate_percentage=0.04, x_translate_prob=0.2,
        blur_percent=0.05, blur_prob=0.12,
        clahe_limit=1.5, clahe_prob=0.2,
        color=True
    )
    

    print('Augmenting data...')

    X_train_augment_1 = []
    for image in X_train:
        aug_img = augmentor_1.augment(image)
        X_train_augment_1.append(aug_img)
    X_train_augment_1 = np.array(X_train_augment_1)

    X_train_augment_2 = []
    for image in X_train:
        aug_img = augmentor_2.augment(image)
        X_train_augment_2.append(aug_img)
    X_train_augment_2 = np.array(X_train_augment_2)

    X_train_augment_3 = []
    for image in X_train:
        aug_img = augmentor_3.augment(image)
        X_train_augment_3.append(aug_img)
    X_train_augment_3 = np.array(X_train_augment_3)


    save_index = 55
    img_bgr = cv2.cvtColor(X_train[save_index], cv2.COLOR_RGB2BGR)
    cv2.imwrite("unaugmented_image.jpg", img_bgr)  
    img_bgr = cv2.cvtColor(X_train_augment_1[save_index], cv2.COLOR_RGB2BGR)
    cv2.imwrite("augment_1_image.jpg", img_bgr)

    # Create triplets (anchor, positive, negative)
    print("Generating triplets")
    all_train_triplets = create_triplets(y_train, num_triplets=15000)
    all_test_triplets = create_triplets(y_test, num_triplets=15000)
    print(f"Generated {len(all_train_triplets)} triplets")
    train_triplets, val_triplets = train_test_split(all_train_triplets, test_size=0.1)



    # Train model

    print("\nTraining model 1 with no augmentor")
    embedding_model_1, contrast_net, val_triplets, all_train_triplets = train_contrastive_model(
                                                                X_train, y_train, 
                                                                train_triplets, val_triplets, 
                                                                batch_size=batch_size, epochs=epochs
                                                            )
    

    print("\nTraining model 2 with augmentor 1")
    embedding_model_2, contrast_net, val_triplets, all_train_triplets = train_contrastive_model(
                                                                X_train_augment_1, y_train, 
                                                                train_triplets, val_triplets, 
                                                                batch_size=batch_size, epochs=epochs
                                                            )

    print("\nTraining model 3 with augmentor 2")
    embedding_model_3, contrast_net, val_triplets, all_train_triplets = train_contrastive_model(
                                                                X_train_augment_2, y_train,
                                                                train_triplets, val_triplets,
                                                                batch_size=batch_size, epochs=epochs
                                                            )

    print("\nTraining model 4 with augmentor 3")
    embedding_model_4, contrast_net, val_triplets, all_train_triplets = train_contrastive_model(
                                                                X_train_augment_3, y_train,
                                                                train_triplets, val_triplets,
                                                                batch_size=batch_size, epochs=epochs
                                                            )

    #  Evaluate model
    #print("Performance on validation set")
    #threshold = evaluate_contrastive_model(embedding_model, val_triplets, X_train, y_train, target_names)
    #  Evaluate model
    #print("Performance on validation set")
    #threshold = evaluate_contrastive_model(embedding_model, val_triplets, X_train, y_train, target_names)


    print("\nModel 1: (no augment) Performance on test set")
    threshold = evaluate_contrastive_model(embedding_model_1, all_test_triplets, X_test, y_test, target_names)
    print("\nModel 2: (augmentor 1) Performance on test set")
    threshold = evaluate_contrastive_model(embedding_model_2, all_test_triplets, X_test, y_test, target_names)
    print("\nModel 3: (augmentor 2) Performance on test set")
    threshold = evaluate_contrastive_model(embedding_model_3, all_test_triplets, X_test, y_test, target_names)
    print("\nModel 4: (augmentor 3) Performance on test set")
    threshold = evaluate_contrastive_model(embedding_model_4, all_test_triplets, X_test, y_test, target_names)