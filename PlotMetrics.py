from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc


def plot_face_pairs(embedding_model, pairs, pair_truth, images, labels, target_names, threshold=1.0):
    """
    Plot face pairs and their embeddings alone with whether the model predicts correctly.
    :param embedding_model: Model which will be used to predict embeddings
    :param pairs: List of pairs shape (n_pairs, 2) containing indexes
    :param pair_truth: List of truth labels shape (n_pairs) containing ground truth labels
    :param images: List of images shape (n, h, w, c)
    :param labels: List of labels is reference to person
    :param target_names: List of names
    :param threshold: Threshold to use in prediction
    :return:
    """
    assert embedding_model.output_shape[-1] == 64, "Embedding size must be 64 for 8x8 heatmap"
    # Make some same-person and some different-person pairs
    n_total = len(pairs)
    fig, axes = plt.subplots(n_total, 6, figsize=(16, 2.2 * n_total))

    for i, ((i1, i2), same) in enumerate(zip(pairs, pair_truth)):
        # Faces
        axes[i, 0].imshow(images[i1].squeeze(), cmap='gray', interpolation='nearest')
        axes[i, 0].set_title(target_names[labels[i1]], fontsize=12)
        axes[i, 0].axis('off')

        axes[i, 1].imshow(images[i2].squeeze(), cmap='gray', interpolation='nearest')
        axes[i, 1].set_title(target_names[labels[i2]], fontsize=12)
        axes[i, 1].axis('off')

        # Embeddings
        e1 = embedding_model.predict(images[i1][None], verbose=0)[0]
        e2 = embedding_model.predict(images[i2][None], verbose=0)[0]
        heat1 = e1.reshape(8, 8)
        heat2 = e2.reshape(8, 8)
        diff = np.abs(e1 - e2).reshape(8, 8)

        # Face 1 heatmap
        axes[i, 2].imshow(heat1, cmap='viridis', vmin=-0.6, vmax=0.6, interpolation='nearest')
        axes[i, 2].axis('off')
        # Face 2 heatmap
        axes[i, 3].imshow(heat2, cmap='viridis', vmin=-0.6, vmax=0.6, interpolation='nearest')
        axes[i, 3].axis('off')
        # Absolute diff heatmap (same scale for all rows)
        axes[i, 4].imshow(diff, cmap='coolwarm', vmin=0, vmax=0.6, interpolation='nearest')
        axes[i, 4].axis('off')

        # Annotate if same/different
        if same:
            axes[i, 1].set_ylabel('SAME', color='green', fontsize=10, rotation=0, labelpad=30, weight='bold')
        else:
            axes[i, 1].set_ylabel('DIFF', color='red', fontsize=10, rotation=0, labelpad=30, weight='bold')

        # Compute distance and prediction
        dist = np.linalg.norm(e1 - e2)
        pred_same = dist < threshold
        pred_str = f"Pred:\n{'SAME' if pred_same else 'DIFF'}\nDist: {dist:.2f}"
        correct = pred_same == same
        pred_color = 'green' if correct else 'red'
        axes[i, 5].text(0.5, 0.5, pred_str, fontsize=12,
                        ha='center', va='center', color=pred_color, weight='bold')
        axes[i, 5].axis('off')
        for j in range(6):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])

    # Column titles
    axes[0, 2].set_title("Face 1\nHeatmap", fontsize=12, fontweight='bold')
    axes[0, 3].set_title("Face 2\nHeatmap", fontsize=12, fontweight='bold')
    axes[0, 4].set_title("Abs Heatmap\nDiff", fontsize=12, fontweight='bold')
    axes[0, 5].set_title("Prediction Threshold: " + str(threshold), fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.show()

def evaluate_verification(embedding_model, pairs, pair_truth, images, verbose=True):
    """
    This function exists to plot and display evaluation results
    :param embedding_model: Model which will be used to predict embeddings
    :param pairs: List of pairs shape (n_pairs, 2) containing indexes
    :param pair_truth: List of truth labels shape (n_pairs) containing ground truth labels
    :param images: List of images shape (n, h, w, c)
    :param verbose: Print additional stats
    :return: Best threshold to use for prediction
    """
    X_test = []
    emb_labels = []
    sub_dict={}
    counter=0
    for pair in pairs:
        if pair[0] not in sub_dict.keys():
            X_test.append(images[pair[0]])
            emb_labels.append(pair[0])
            sub_dict[pair[0]] = counter
            counter += 1
        if pair[1] not in sub_dict.keys():
            X_test.append(images[pair[1]])
            emb_labels.append(pair[1])
            sub_dict[pair[1]] = counter
            counter += 1
    # Compute embeddings
    X_test= np.array(X_test)
    emb = embedding_model.predict(X_test, batch_size=128, verbose=0)
    plot_pca_tsne(emb, emb_labels)
    # Compute distances
    distances = np.array([np.linalg.norm(emb[sub_dict[a]] - emb[sub_dict[b]]) for a, b in pairs])

    # ROC curve & AUC
    fpr, tpr, thresholds = roc_curve(pair_truth, -distances)  # minus sign: lower distance = more similar
    roc_auc = auc(fpr, tpr)
    best_idx = np.argmax(tpr - fpr)
    best_threshold = -thresholds[best_idx]
    y_pred = (distances < best_threshold).astype(int)

    # Metrics
    acc = accuracy_score(pair_truth, y_pred)
    prec = precision_score(pair_truth, y_pred)
    rec = recall_score(pair_truth, y_pred)
    f1 = f1_score(pair_truth, y_pred)

    # Plot ROC
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Verification ROC Curve')
    plt.legend()
    plt.show()

    if verbose:
        print(f"Accuracy: {acc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}, ROC AUC: {roc_auc:.3f}")
        print(f"Best threshold (distance): {best_threshold:.3f}")

    return best_threshold


def plot_pca_tsne(emb, labels, class_names=None, n_samples=500):
    """
    Sub function for plotting PCA and t-SNE
    :param emb: array of embedding matrices
    :param labels: label reference to person
    :param class_names: Constructing legend
    :param n_samples: Samples to use for
    :return:
    """
    # Subsample for clarity
    if emb.shape[0] > n_samples:
        idxs = np.random.choice(emb.shape[0], n_samples, replace=False)
        emb_plot = emb[idxs]
        labels_plot = np.array(labels)[idxs]
    else:
        emb_plot = emb
        labels_plot = np.array(labels)

    # PCA
    pca = PCA(n_components=2)
    pca_2d = pca.fit_transform(emb_plot)

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=30)
    tsne_2d = tsne.fit_transform(emb_plot)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, data, title in zip(axes, [pca_2d, tsne_2d], ["PCA of Embeddings", "t-SNE of Embeddings"]):
        scatter = ax.scatter(data[:, 0], data[:, 1], c=labels_plot, cmap='tab20', s=18, alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
    if class_names is not None and len(set(labels_plot)) < 20:
        handles, _ = scatter.legend_elements()
        axes[1].legend(handles, [class_names[i] for i in set(labels_plot)], title="Classes", bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.show()

def plot_training_history(history):
    """
    Plots the loss curves and Accuracy curves
    :param history: History object from training
    :return:
    """
    history_dict = history.history

    # Check which plots we have
    has_loss = 'loss' in history_dict or 'val_loss' in history_dict
    has_accuracy = 'accuracy' in history_dict or 'val_accuracy' in history_dict

    if not has_loss and not has_accuracy:
        print("No loss or accuracy data found to plot.")
        return

    # Create subplots
    ncols = has_loss + has_accuracy
    fig, axs = plt.subplots(1, ncols, figsize=(6 * ncols, 5))
    if ncols == 1:
        axs = [axs]  # make axs always iterable
    plot_idx = 0

    # Plot Loss
    if has_loss:
        ax = axs[plot_idx]
        if 'loss' in history_dict:
            ax.plot(history_dict['loss'], label='Train Loss')
        if 'val_loss' in history_dict:
            ax.plot(history_dict['val_loss'], label='Validation Loss')
        ax.set_title('Loss Curve')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True)
        plot_idx += 1

    # Plot Accuracy
    if has_accuracy:
        ax = axs[plot_idx]
        if 'accuracy' in history_dict:
            ax.plot(history_dict['accuracy'], label='Train Accuracy')
        if 'val_accuracy' in history_dict:
            ax.plot(history_dict['val_accuracy'], label='Validation Accuracy')
        ax.set_title('Accuracy Curve')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()