import tensorflow as tf
from sklearn.model_selection import train_test_split


from DataPipeline import get_data_image, get_data_person, FaceImageAugmentor
from Generators import *
from PlotMetrics import *
from Models import *

def main():
    # Load Data
    tf.keras.backend.clear_session()
    #policy = tf.keras.mixed_precision.Policy('mixed_float16')
    #tf.keras.mixed_precision.set_global_policy(policy)
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
    X_train, y_train, X_test, y_test, target_names = get_data_image(size=(160,160),color=True,min_pics=2)
    #X_train, y_train = augmentor.augment_batch(X_train, y_train)
    X_train = np.array(X_train, dtype=np.float32)/255.0
    X_test = np.array(X_test, dtype=np.float32)/255.0

    #X_test = X_test *2 -1
    #X_train = X_train *2 -1
    # Create triplets (anchor, positive, negative)

    print("Generating triplets")
    train_triplets = create_triplets(y_train, num_triplets=6000)
    test_triplets = create_triplets(y_test, num_triplets=1000)
    print(f"Generated {len(train_triplets)} triplets")

    # Train/test split for triplets
    batch_size = 64
    train_gen = triplet_generator(X_train, train_triplets, batch_size)
    test_gen = triplet_generator(X_test, test_triplets, batch_size)
    steps_per_epoch = len(train_triplets) // batch_size
    validation_steps = len(test_triplets) // batch_size

    # Train model
    print("Training model")
    full_network, embedding_model = compile_simple_triplet(X_train.shape[1:])
    embedding_model.summary()
    history = full_network.fit(train_gen,epochs=15,steps_per_epoch=steps_per_epoch,validation_data=test_gen,validation_steps=validation_steps)
    plot_training_history(history,"Triplet")

    # Visualize embeddings
    print("Evaluating model")
    pairs, pair_labels = pairs_from_triplets(test_triplets)
    summary = evaluate_verification(embedding_model, pairs, pair_labels, X_test,plot=True)
    pairs, pair_labels = pairs_from_triplets(test_triplets, n_pairs=8, other=True)
    plot_face_pairs(embedding_model, pairs, pair_labels, X_test, y_test, target_names,threshold=summary["threshold"])
    tf.keras.backend.clear_session()


if __name__ == "__main__":
    main()