import tensorflow as tf
from tensorflow.keras import layers, Model, Input
import tensorflow.keras.backend as K
from FaceNetBase import InceptionResNetV1


def contrastive_loss(y_true, y_pred):
    margin = 0.5
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

def triplet_loss(margin=0.2):
    def loss(y_true, y_pred):
        anchor, positive, negative = y_pred[:, 0, :], y_pred[:, 1, :], y_pred[:, 2, :]

        pos_dist = K.sum(K.square(anchor - positive), axis=1)
        neg_dist = K.sum(K.square(anchor - negative), axis=1)

        return K.mean(K.maximum(pos_dist - neg_dist + margin, 0.0))
    return loss

def build_facenet_model(base_path, embedding_dim=128, input_shape=(160,160,3), freeze_base=True):
    base_model = InceptionResNetV1(weights_path=base_path)
    if freeze_base:
        base_model.trainable = False

    inputs = Input(shape=input_shape)
    x = base_model(inputs)
    x = layers.Dense(embedding_dim)(x)
    x = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(x)
    return Model(inputs, x, name="EmbeddingModel")

def build_simple_model(input_shape):
    embedding_model = tf.keras.Sequential([
        layers.Input(shape=input_shape),
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
    return embedding_model


def resnet_block(x, filters, strides=1):
    shortcut = x

    x = layers.Conv2D(filters, 3, strides=strides, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters, 3, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)


    if strides != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=strides, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    return x


def build_deep_model(input_shape):
    input = Input(shape=input_shape)
    x= layers.Conv2D(64, 7, strides=2, padding='same')(input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
    x = resnet_block(x, 64, 1)
    x = resnet_block(x, 64, 1)
    x = resnet_block(x, 64, 1)
    x = resnet_block(x, 128, 2)
    x = resnet_block(x, 128, 1)
    x = resnet_block(x, 256, 2)
    x = resnet_block(x, 256, 1)
    x = resnet_block(x, 512, 2)
    x = resnet_block(x, 512, 1)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64)(x)
    x = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(x)
    model = Model(inputs=input, outputs=x)
    return model

def compile_triplet_internal(embedding_model,input_shape):
    # Build Siamese network
    input_anchor = Input(shape=input_shape)
    input_positive = Input(shape=input_shape)
    input_negative = Input(shape=input_shape)

    embedding_anchor = embedding_model(input_anchor)
    embedding_positive = embedding_model(input_positive)
    embedding_negative = embedding_model(input_negative)

    merged_output = layers.Lambda(lambda x: tf.stack(x, axis=1))(
        [embedding_anchor, embedding_positive, embedding_negative]
    )
    triplet_net = Model([input_anchor, input_positive, input_negative], merged_output)
    triplet_net.compile(loss=triplet_loss(), optimizer='adam')
    return triplet_net, embedding_model

def compile_contrastive_internal(embedding_model,input_shape):
    # Build Siamese network
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    embedding_a = embedding_model(input_a)
    embedding_b = embedding_model(input_b)

    merged_output = layers.Lambda(lambda x: tf.sqrt(tf.reduce_sum(tf.square(x[0] - x[1]), axis=1, keepdims=True)))(
        [embedding_a, embedding_b]
    )
    contrast_net = Model([input_a, input_b], merged_output)
    contrast_net.compile(loss=contrastive_loss, optimizer='adam')
    return contrast_net, embedding_model

def compile_facenet(input_shape):
    embedding_model = build_facenet_model("facenet_keras_weights.h5", embedding_dim=64, input_shape=input_shape)
    return compile_triplet_internal(embedding_model,input_shape)

def compile_simple_triplet(input_shape):
    embedding_model = build_simple_model(input_shape)
    return compile_triplet_internal(embedding_model,input_shape)

def compile_deep_triplet(input_shape):
    embedding_model = build_deep_model(input_shape)
    return compile_triplet_internal(embedding_model,input_shape)

def compile_simple_contrastive(input_shape):
    embedding_model = build_simple_model(input_shape)
    return compile_contrastive_internal(embedding_model,input_shape)

def compile_deep_contrastive(input_shape):
    embedding_model = build_deep_model(input_shape)
    return compile_contrastive_internal(embedding_model,input_shape)