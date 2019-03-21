import numpy as np
import random

random.seed(100)


def generate_batch(pairs, n_positive=50, negative_ratio=1.0, classification = False):
    """Generate batches of samples for training.
       Random select positive samples
       from pairs and randomly select negatives."""
    global animes
    global tags
    global pairs_set
    global neg_label

    # Create empty array to hold batch
    batch_size = n_positive * (1 + negative_ratio)
    batch = np.zeros((int(batch_size), 3))

    # Adjust label based on task
    if classification:
        neg_label = 0
    else:
        neg_label = -1

    # Continue to yield samples
    while True:
        # Randomly choose positive examples
        for idx, (book_id, link_id) in enumerate(random.sample(pairs, n_positive)):
            batch[idx, :] = (book_id, link_id, 1)
        print(idx)
        idx += 1

        # Add negative examples until reach batch size
        while idx < batch_size:

            # Random selection
            random_anime = random.randrange(len(animes))
            random_tag = random.randrange(len(tags))

            # Check to make sure this is not a positive example
            if (random_anime, random_tag) not in pairs_set:
                # Add to batch and increment index
                batch[idx, :] = (animes[random_anime]['anime_id'], random_tag, neg_label)
                idx += 1

        # Make sure to shuffle order
        np.random.shuffle(batch)
        yield {'animes': batch[:, 0], 'tag': batch[:, 1]}, batch[:, 2]
