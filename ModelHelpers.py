from Generators import *
from Models import compile_simple_contrastive, compile_deep_contrastive, compile_simple_triplet, compile_deep_triplet, compile_facenet
from PlotMetrics import *

def train_contrastive_model(net,X, train_triplets, test_triplets, batch_size=64, epochs=10, plot=None):
    """
        Builds and trains the contrastive (Siamese) model.
        Returns the trained embedding model and the contrastive network.
    """
    # Train model
    history = net.fit(
        pair_generator(X, train_triplets, batch_size),
        epochs=epochs,
        steps_per_epoch= len(train_triplets) // batch_size,
        validation_data= pair_generator(X, test_triplets, batch_size),
        validation_steps= len(test_triplets) // batch_size
    )
    if plot:
        plot_training_history(history, plot)

def train_triplet_model(net,X, train_triplets, test_triplets, batch_size=64, epochs=10, plot=None):
    # Train model
    history = net.fit(
        triplet_generator(X, train_triplets, batch_size),
        epochs=epochs,
        steps_per_epoch=len(train_triplets) // batch_size,
        validation_data=triplet_generator(X, test_triplets, batch_size),
        validation_steps=len(test_triplets) // batch_size
    )
    if plot is not None:
        plot_training_history(history, plot)

# 5 different 1 line function signatures
def train_facenet(X, train_triplets, test_triplets, batch_size=32, epochs=10, plot=None):
    triplet_net, embedding_model = compile_facenet(X.shape[1:])
    train_triplet_model(triplet_net, X, train_triplets, test_triplets, batch_size, epochs//2, plot=plot)
    return embedding_model, triplet_net

def train_simple_triplet(X, train_triplets, test_triplets, batch_size=32, epochs=10, plot=None):
    triplet_net, embedding_model = compile_simple_triplet(X.shape[1:])
    train_triplet_model(triplet_net, X, train_triplets, test_triplets, batch_size, epochs, plot=plot)
    return embedding_model, triplet_net

def train_deep_triplet(X, train_triplets, test_triplets, batch_size=32, epochs=10, plot=None):
    triplet_net, embedding_model = compile_deep_triplet(X.shape[1:])
    train_triplet_model(triplet_net, X, train_triplets, test_triplets, batch_size, epochs, plot=plot)
    return embedding_model, triplet_net

def train_simple_contrastive(X, train_triplets, test_triplets, batch_size=32, epochs=10, plot=None):
    triplet_net, embedding_model = compile_simple_contrastive(X.shape[1:])
    train_contrastive_model(triplet_net, X, train_triplets, test_triplets, batch_size, epochs, plot=plot)
    return embedding_model, triplet_net

def train_deep_contrastive(X, train_triplets, test_triplets, batch_size=32, epochs=10, plot=None):
    triplet_net, embedding_model = compile_deep_contrastive(X.shape[1:])
    train_contrastive_model(triplet_net, X, train_triplets, test_triplets, batch_size, epochs, plot=plot)
    return embedding_model, triplet_net


def evaluate_model(embedding_model, test_triplets, X, y, target_names,verbose=False,plot=False):
    """
        Evaluates the trained embedding model and plots results.
    """
    print("Evaluating model")
    pairs, pair_labels = pairs_from_triplets(test_triplets)
    summary = evaluate_verification(embedding_model, pairs, pair_labels, X,verbose=verbose,plot=plot)
    if plot:
        pairs, pair_labels = pairs_from_triplets(test_triplets, n_pairs=8, other=True)
        plot_face_pairs(embedding_model, pairs, pair_labels, X, y, target_names, threshold=summary["threshold"])
    return summary