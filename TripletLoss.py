import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from sklearn.model_selection import train_test_split
import tensorflow.keras.backend as K
import visualkeras

from Generators import *
from PlotMetrics import *
from DataPipeline import get_data, FaceImageAugmentor

def triplet_loss(margin=1.0):
    def loss(y_true, y_pred):
        anchor, positive, negative = y_pred[:, 0, :], y_pred[:, 1, :], y_pred[:, 2, :]

        pos_dist = K.sum(K.square(anchor - positive), axis=1)
        neg_dist = K.sum(K.square(anchor - negative), axis=1)

        return K.mean(K.maximum(pos_dist - neg_dist + margin, 0.0))
    return loss

def main():
    # Load Data
    tf.keras.backend.clear_session()
    #policy = tf.keras.mixed_precision.Policy('mixed_float16')
    #tf.keras.mixed_precision.set_global_policy(policy)
    augmentor = FaceImageAugmentor(
        number_of_output=2,
        color=True, output_size=(160,160),
        rotation_range=3, rotation_prob=0.01,
        fliplr_prob=0.5,
        brightness_range=(0.7, 1.3), brightness_prob=0.7,
        y_translate_percentage=0.1, y_translate_prob=0.2,
        x_translate_percentage=0.1, x_translate_prob=0.2,
        blur_percent=0.08, blur_prob=0.4,
        clahe_limit=0.8, clahe_prob=0.4
    )
    X, y, target_names = get_data(size=(64,64),color=True, min_pics=2)

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
    input_anchor = Input(shape=X.shape[1:])
    input_positive = Input(shape=X.shape[1:])
    input_negative = Input(shape=X.shape[1:])

    embedding_anchor = embedding_model(input_anchor)
    embedding_positive = embedding_model(input_positive)
    embedding_negative = embedding_model(input_negative)

    merged_output = layers.Lambda(lambda x: tf.stack(x, axis=1))(
        [embedding_anchor, embedding_positive, embedding_negative]
    )
    triplet_net = Model([input_anchor, input_positive, input_negative], merged_output)
    triplet_net.compile(loss=triplet_loss(), optimizer='adam')

    # Train/test split for triplets
    batch_size = 32
    train_triplets, test_triplets = train_test_split(triplets, test_size=0.1)
    train_gen = triplet_generator(X, train_triplets, batch_size)
    test_gen = triplet_generator(X, test_triplets, batch_size)
    steps_per_epoch = len(train_triplets) // batch_size//4
    validation_steps = len(test_triplets) // batch_size//4

    # Train model
    print("Training model")
    H3 = triplet_net.fit(train_gen,epochs=50,steps_per_epoch=steps_per_epoch,validation_data=test_gen,validation_steps=validation_steps)
    plot_training_history(H3)
    # Visualize embeddings
    print("Evaluating model")
    pairs, pair_labels = pairs_from_triplets(test_triplets)
    threshold = evaluate_verification(embedding_model, pairs, pair_labels, X)
    pairs, pair_labels = pairs_from_triplets(test_triplets, n_pairs=8, other=True)
    plot_face_pairs(embedding_model, pairs, pair_labels, X, y, target_names)
    tf.keras.backend.clear_session()


if __name__ == "__main__":
    main()