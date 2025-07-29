import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from sklearn.model_selection import train_test_split
import tensorflow.keras.backend as K

from Generators import *
from PlotMetrics import *
from DataPipeline import get_data
from FaceNetBase import InceptionResNetV1

def triplet_loss(margin=1.0):
    def loss(y_true, y_pred):
        anchor, positive, negative = y_pred[:, 0, :], y_pred[:, 1, :], y_pred[:, 2, :]
        pos_dist = K.sum(K.square(anchor - positive), axis=1)
        neg_dist = K.sum(K.square(anchor - negative), axis=1)
        return K.mean(K.maximum(pos_dist - neg_dist + margin, 0.0))
    return loss

def build_embedding_model(base_path, embedding_dim=128, input_shape=(160,160,3), freeze_base=True):
    base_model = InceptionResNetV1(weights_path=base_path)
    if freeze_base:
        base_model.trainable = False

    inputs = Input(shape=input_shape)
    x = base_model(inputs)
    x = layers.Dense(embedding_dim)(x)
    x = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(x)
    return Model(inputs, x, name="EmbeddingModel")

def main():

    X_img, y, target_names = get_data(size=(160,160), color=True)  # X: (N, 160, 160, 3), RGB format

    # Normalize (expected range for FaceNet is [-1, 1])
    X = (X_img.astype(np.float32)*2.0) - 1.0

    # Create triplets
    print("Generating triplets")
    triplets = create_triplets(y, num_triplets=15000)
    print(f"Generated {len(triplets)} triplets")

    # Load pretrained model + attach embedding head
    embedding_model = build_embedding_model("facenet_keras_weights.h5", embedding_dim=64, input_shape=X.shape[1:])
    embedding_model.summary()

    # Build Siamese network
    input_anchor = Input(shape=X.shape[1:])
    input_positive = Input(shape=X.shape[1:])
    input_negative = Input(shape=X.shape[1:])
    anchor_emb = embedding_model(input_anchor)
    pos_emb = embedding_model(input_positive)
    neg_emb = embedding_model(input_negative)

    merged_output = layers.Lambda(lambda x: tf.stack(x, axis=1))([anchor_emb, pos_emb, neg_emb])
    triplet_net = Model([input_anchor, input_positive, input_negative], merged_output)
    triplet_net.compile(loss=triplet_loss(), optimizer='adam')

    # Train/test split for triplets
    batch_size = 64
    train_triplets, test_triplets = train_test_split(triplets, test_size=0.1)
    train_gen = triplet_generator(X, train_triplets, batch_size)
    test_gen = triplet_generator(X, test_triplets, batch_size)
    steps_per_epoch = len(train_triplets) // batch_size
    val_steps = len(test_triplets) // batch_size

    # Train model
    print("Training model")
    triplet_net.fit(train_gen,epochs=20, steps_per_epoch=steps_per_epoch,validation_data=test_gen,validation_steps=val_steps)

    # Visualize and evaluate
    print("Evaluating model")
    pairs, pair_labels = pairs_from_triplets(test_triplets)
    threshold = evaluate_verification(embedding_model, pairs, pair_labels, X)
    pairs, pair_labels = pairs_from_triplets(test_triplets, n_pairs=8, other=True)
    plot_face_pairs(embedding_model, pairs, pair_labels, X_img, y, target_names)

if __name__ == "__main__":
    main()