from model import anime_embedding_model
from find_similar import find_similar
from keras.models import Model
import csv
import json
import random
import numpy as np
import pickle
import triarray
import tensorflow as tf

random.seed(100)

animePath = r'E:\Benutzer\Eigene Dokumente\Waifu Laifu\Recommender\anilist-anime\anime_filtered_new.csv'
tagsPath = r'E:\Benutzer\Eigene Dokumente\Waifu Laifu\Recommender\anilist-anime\tags.json'
genresPath = r'E:\Benutzer\Eigene Dokumente\Waifu Laifu\Recommender\anilist-anime\genres.json'

animes = []
tags = {}
pairs = []
pairs_set = []

anime_index = {}
index_anime = {}

mal_index = {}
anilist_index = {}

tag_index = {}
index_tag = {}

max_popularity = -1


def save_to_file(object, path):
    with open(path, 'wb') as f:
        pickle.dump(object, f, pickle.HIGHEST_PROTOCOL)


def load_from_file(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def init():
    global animes
    global tags
    global pairs
    global pairs_set
    global anime_index
    global index_anime
    global index_tag
    global tag_index
    global max_popularity

    # Load tags and genres
    with open(tagsPath, mode='r', encoding="utf8") as tags_file:
        tags = json.load(tags_file)
        cur = 0
        for tag in tags:
            tags[tag] = cur
            cur += 1
        with open(genresPath, mode='r', encoding="utf8") as genres_file:
            genres = json.load(genres_file)
            for genre in genres:
                if genre not in tags:
                    tags[genre] = cur
                    cur += 1
        tags["ORIGINAL"] = cur
        cur += 1
        tags["MANGA"] = cur
        cur += 1
        tags["LIGHT_NOVEL"] = cur
        cur += 1
        tags["VISUAL_NOVEL"] = cur
        cur += 1
        tags["VIDEO_GAME"] = cur
        cur += 1
        tags["OTHER"] = cur
        cur += 1
        tags["TV"] = cur
        cur += 1
        tags["TV_SHORT"] = cur
        cur += 1
        tags["MOVIE"] = cur
        cur += 1
        tags["SPECIAL"] = cur
        cur += 1
        tags["OVA"] = cur
        cur += 1
        tags["ONA"] = cur
        cur += 1
        tags["MUSIC"] = cur
        tag_index = tags
        index_tag = {idx: tag for tag, idx in tag_index.items()}

    # Load anime
    with open(animePath, mode='r', encoding="utf8") as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            new_genres = row['genres'].split(', ')
            row['genres'] = new_genres
            new_tags = row['tags'].split(', ')
            final_tags = []
            for tag in new_tags:
                if tag == '':
                    continue
                split = tag.split(',')
                final_tags.append((split[0][1:], split[1][:-1]))
            row['tags'] = final_tags
            while row['title'] in anime_index:
                row['title'] = row['title'] + '.'
            row['popularity'] = int(row['popularity'])
            row['score'] = int(row['score'])
            animes.append(row)
            anime_index[row['title']] = line_count
            index_anime[line_count] = row['title']
            mal_index[row['mal_id']] = line_count
            anilist_index[row['anime_id']] = line_count
            line_count += 1
            max_popularity = max(max_popularity, int(row['popularity']))
        print(f'Processed {line_count} lines.')

    # Get pairs
    cur_id = 0
    for anime in animes:
        for genre in anime['genres']:
            if genre != "" and genre != "None":
                pairs.append((cur_id, (tags[genre], 100)))
        for tag in anime['tags']:
            if tag != "" and tag != "None":
                pairs.append((cur_id, (tags[tag[0]], int(tag[1]))))
        if anime['type'] != 'None':
            pairs.append((cur_id, (tags[anime['type']], 100)))
        if anime['source'] != 'None':
            pairs.append((cur_id, (tags[anime['source']], 100)))
        cur_id += 1
    pairs_set = set(pairs)


def generate_batch(pairs, n_positive=50, negative_ratio=1.0, classification = False, use_rank = True):
    """Generate batches of samples for training.
       Random select positive samples
       from pairs and randomly select negatives."""
    global animes
    global tags
    global pairs_set
    global neg_label
    global max_popularity

    # Create empty array to hold batch
    batch_size = n_positive * (1 + negative_ratio)
    batch = np.zeros((int(batch_size), 5))

    # Adjust label based on task
    if classification:
        neg_label = 0
    else:
        neg_label = -1

    # Continue to yield samples
    while True:
        # Randomly choose positive examples
        for idx, (anime_id, tag_pair) in enumerate(random.sample(pairs, n_positive)):
            score = (animes[anime_id]['score']/100 - 0.5) * 2
            popularity = (animes[anime_id]['popularity']/max_popularity - 1) * 2
            rank = 1
            if use_rank:
                rank = (tag_pair[1] / 100 - 0.5) * 2
            batch[idx, :] = (anime_id, tag_pair[0], rank, score, popularity)
        idx += 1

        # Add negative examples until reach batch size
        while idx < batch_size:

            score = (float(animes[anime_id]['score'])/100 - 0.5) * 2
            popularity = (float(animes[anime_id]['popularity'])/max_popularity - 0.5) * 2

            # Random selection
            random_anime = random.randrange(len(animes))
            random_tag = random.randrange(len(tags))

            # Check to make sure this is not a positive example
            if (random_anime, random_tag) not in pairs_set:
                # Add to batch and increment index
                batch[idx, :] = (random_anime, random_tag, neg_label, score, popularity)
                idx += 1

        # Make sure to shuffle order
        np.random.shuffle(batch)
        yield {'anime': batch[:, 0], 'tag': batch[:, 1]}, {'tag_output': batch[:, 2], 'score_output': batch[:, 3], 'popularity_output': batch[:, 4]}


def extract_weights(name, model):
    """Extract weights from a neural network model"""

    # Extract weights
    weight_layer = model.get_layer(name)
    weights = weight_layer.get_weights()[0]

    # Normalize
    weights = weights / np.linalg.norm(weights, axis=1).reshape((-1, 1))
    return weights


def save_all():
    global animes
    global anime_index
    global index_anime
    global mal_index
    global anilist_index
    save_to_file(mal_index, '../../resources/tag_based/mal_index.pkl')
    save_to_file(anilist_index, '../../resources/tag_based/anilist_index.pkl')
    save_to_file(animes, '../../resources/tag_based/animes.pkl')
    save_to_file(anime_index, '../../resources/tag_based/anime_index.pkl')
    save_to_file(index_anime, '../../resources/tag_based/index_anime.pkl')


def load_all():
    global animes
    global anime_index
    global index_anime
    animes = load_from_file('../../resources/tag_based/animes.pkl')
    anime_index = load_from_file('../../resources/tag_based/anime_index.pkl')
    index_anime = load_from_file('../../resources/tag_based/index_anime.pkl')


def save_similarity_matrix():
    global animes
    global anime_index
    global index_anime
    global anime_weights
    global tag_index
    global index_tag

    matrix = np.zeros((len(animes),len(animes)))
    for i in range(len(animes)):
        anime = animes[i]
        dists, _ = find_similar(anime['title'], anime_weights, anime_index, index_anime, tag_index, index_tag, n=len(animes), return_dist=True)
        matrix[i] = dists
    trimatrix = triarray.matrix_to_tri(matrix)
    np.save('../../resources/tag_based/trimatrix.npy', trimatrix)


init()
model = anime_embedding_model(anime_index, tag_index, embedding_size=30)
model.summary()

train = False

if train:
    n_positive = 1024

    gen = generate_batch(pairs, n_positive, negative_ratio=2)

    # Train
    h = model.fit_generator(gen, epochs=30,
                            steps_per_epoch=len(pairs) // n_positive,
                            verbose=2)
    model.save('models/tag_based_extended.h5')
else:
    model.load_weights('models/tag_based_extended.h5')

# Extract embeddings
anime_weights = extract_weights('anime_embedding', model)
tag_weights = extract_weights('tag_embedding', model)
find_similar('Shingeki no Kyojin', anime_weights, anime_index, index_anime, tag_index, index_tag, n=20)
save_all()
save_similarity_matrix()
