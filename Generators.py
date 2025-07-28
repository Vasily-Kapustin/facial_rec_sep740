import numpy as np


def create_triplets(y, num_triplets=10000):
    """
    Randomly create triplets from the labels
    Index inside triplet referring to array of images that accompanies the labels
    :param y: Array of labels
    :param num_triplets: how many triplets to generate
    :return: Triplets tuple shape (index,index,index)
    """
    # We want triplets to be randomly distributed by person not by image
    person_photo_dict = {}
    for idx, label in enumerate(y):
        if label not in person_photo_dict.keys():
            person_photo_dict[label] = [idx]
        else:
            person_photo_dict[label].append(idx)
    triplets = np.zeros((num_triplets, 3),dtype=int)
    key_list = list(person_photo_dict.keys())
    for i in range(num_triplets):
        anchor_person = np.random.choice(key_list)
        negative_person = anchor_person
        while negative_person == anchor_person: #make sure that anchor and negative people are not the same person
            negative_person = np.random.choice(key_list)
        a, p = np.random.choice(person_photo_dict[anchor_person], 2, replace=False)
        n = np.random.choice(person_photo_dict[negative_person])
        triplets[i] = [a, p, n]
    return triplets

def triplet_generator(X, triplets, batch_size=32):
    """
    https://www.tensorflow.org/guide/data
    This is a generator function it allows images to be loaded, this doesn't really matter on jupyter, but it does on my pc
    Ground truth already encoded in triplet to we zero our label
    :param X: np.array of Images shape (n, w, h, c)
    :param triplets:  Triplet tuple
    :param batch_size: Batch size
    :return: np.array of Images shape [n, 3, w, h, c], np.array of Labels shape [n]
    """
    while True:
        np.random.shuffle(triplets)
        for i in range(0, len(triplets) - batch_size, batch_size): #subdivide set into batches
            batch = triplets[i:i+batch_size]
            anchor = np.array([X[a] for a,_,_ in batch])
            positive = np.array([X[p] for _,p,_ in batch])
            negative = np.array([X[n] for _,_,n in batch])
            yield (anchor, positive, negative), np.zeros((batch_size, 1))

def pair_generator(X, triplets, batch_size=32):
    """
    Pairs are made from doing 50-50 random on a triplet, ground truth must be encoded because it's a pair
    This function is to feed into contrastive loss training
    :param X: np.array of Images shape (n, w, h, c)
    :param triplets:  Triplet tuple
    :param batch_size: Batch size
    :return: np.array of Images shape (n, 2, w, h, c), np.array of Labels shape [n]
    """
    while True:
        np.random.shuffle(triplets)
        for i in range(0, len(triplets) - batch_size, batch_size):
            batch = triplets[i:i+batch_size]
            a=[]
            b=[]
            l=np.zeros((batch_size, 1))
            for j in range(len(batch)):
                first, second, third = batch[j]
                a.append(X[first])
                if np.random.random() < 0.5:
                    b.append(X[second])
                    l[j] = 1.0
                else:
                    b.append(X[third])
                    l[j] = 0.0
            yield (np.array(a), np.array(b)), l


def pairs_from_triplets(triplets, n_pairs=None, other = False):
    """
    This function is for our own use when we need to evaluate pairs
    :param triplets: Triplet tuple
    :param n_pairs: None means generate all pairs
    :param other: whether 1 triplet becomes 2 pairs or 1 pair
    :return: np.array of pairs shape (n, 2), np.array of labels shape (n)
    """
    pairs = []
    labels = []
    sel_triplets = triplets

    if n_pairs is not None:
        if not other:
            n_pairs = int(n_pairs/2.0)
        sel_triplets = triplets[np.random.choice(triplets.shape[0], n_pairs, replace=False)]
    if other:
        for triplet in sel_triplets:
            pick = np.random.randint(1,3)
            pairs.append((triplet[0], triplet[pick]))
            labels.append(2-pick)
    else:
        for triplet in sel_triplets:
            pairs.append((triplet[0], triplet[1]))
            pairs.append((triplet[0], triplet[2]))
            labels.append(1)
            labels.append(0)
    return np.array(pairs), np.array(labels)