import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
import tensorflow.keras.backend as K

from Generators import *
from PlotMetrics import *
from DataPipeline import get_data, FaceImageAugmentor



def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

def euclidean_distance(vectors):
    x, y = vectors
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

def main():
    # Load LFW dataset
    augmentor = FaceImageAugmentor(
        number_of_output=5,

        rotation_range=3, rotation_prob=0.01,
        fliplr_prob=0.5,
        brightness_range=(0.7, 1.3), brightness_prob=0.7,
        y_translate_percentage=0.1, y_translate_prob=0.2,
        x_translate_percentage=0.1, x_translate_prob=0.2,
        blur_percent=0.08, blur_prob=0.4,
        clahe_limit=0.8, clahe_prob=0.4
    )

    X, y, target_names = get_data( min_pics=2)

    # Create triplets (anchor, positive, negative)
    print("Generating triplets")
    triplets = create_triplets(y, num_triplets=15000)
    print(f"Generated {len(triplets)} triplets")

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
    input_a= Input(shape=X.shape[1:])
    input_b= Input(shape=X.shape[1:])

    embedding_a= embedding_model(input_a)
    embedding_b = embedding_model(input_b)

    merged_output = layers.Lambda(lambda x: tf.sqrt(tf.reduce_sum(tf.square(x[0] - x[1]), axis=1, keepdims=True)))(
        [embedding_a, embedding_b]
    )
    contrast_net = Model([input_a, input_b], merged_output)
    contrast_net.compile(loss=contrastive_loss, optimizer='adam')

    # Train/test split for triplets
    batch_size = 256
    train_triplets, test_triplets = train_test_split(triplets, test_size=0.1)
    train_gen = pair_generator(X, train_triplets, batch_size)
    test_gen = pair_generator(X, test_triplets, batch_size)
    steps_per_epoch = len(train_triplets) // batch_size
    validation_steps = len(test_triplets) // batch_size

    # Train model
    print("Training model")
    H2= contrast_net.fit(train_gen,epochs=40,steps_per_epoch=steps_per_epoch,validation_data=test_gen,validation_steps=validation_steps)
    plot_training_history(H2)
    # Visualize and evaluate
    print("Evaluating model")
    pairs, pair_labels = pairs_from_triplets(test_triplets)
    threshold = evaluate_verification(embedding_model, pairs, pair_labels, X)
    pairs, pair_labels = pairs_from_triplets(triplets, n_pairs=8, other=True)
    plot_face_pairs(embedding_model, pairs, pair_labels, X, y, target_names,threshold=0.5)



if __name__ == "__main__":
    main()