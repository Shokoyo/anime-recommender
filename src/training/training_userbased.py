from model_userbased import anime_embedding_model
from find_similar import find_similar
from keras.models import Model
import csv
import pickle
import random
import numpy as np

random.seed(100)

animePath = r'E:\Benutzer\Eigene Dokumente\Waifu Laifu\Recommender\anilist-anime\anime_filtered_new.csv'
usersPath = r'E:\Benutzer\Eigene Dokumente\Waifu Laifu\Recommender\myanimelist\users_cleaned.csv'
listsPath = r'E:\Benutzer\Eigene Dokumente\Waifu Laifu\Recommender\myanimelist\animelists_cleaned.csv'

animes = []
users = []
triplets = []
triplets_set = []

anime_index = {}
index_anime = {}
# map myanimelist id to array index
mal_index_anime = {}

user_index = {}
index_user = {}

rated_anime = {}


def init():
    global animes
    global triplets
    global triplets_set
    global anime_index
    global index_anime
    global index_user
    global user_index
    global rated_anime
    global users

    # Load anime
    with open(animePath, mode='r', encoding="utf8") as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            new_genres = row['genres'].split(', ')
            row['genres'] = new_genres
            new_tags = row['tags'].split(', ')
            row['tags'] = new_tags
            animes.append(row)
            while row['title'] in anime_index:
                row['title'] = row['title'] + '.'
            anime_index[row['title']] = line_count
            index_anime[line_count] = row['title']
            line_count += 1
        print(f'Processed {line_count} lines.')
        print(anime_index)
        print(index_anime)

    # Load user data
    # Store users as dict containing name, id, mean score, watch list
    with open(usersPath, mode='r', encoding="utf8") as users_file:
        csv_reader = csv.DictReader(users_file)
        user_count = 0
        for row in csv_reader:
            if row['username'] not in user_index:
                user_index[row['username']] = user_count
                index_user[user_count] = row['username']
                user_count += 1
                users.append({
                    'username': row['username'],
                    'user_id': row['user_id'],
                    'stats_mean_score': row['stats_mean_score'],
                    'watchlist': []
                })

    # Create watchlists
    with open(listsPath, mode='r', encoding="utf8") as lists_file:
        csv_reader = csv.DictReader(lists_file)
        for row in csv_reader:
            index = user_index[row['username']]
            users[index]['watchlist'].append((row['anime_id'], row['my_score']))
            rated_anime[row['anime_id']] = True

    users = [user for user in users if len(user['watchlist']) <= 30]
    user_index = {}
    index_user = {}

    new_anime = [anime for anime in animes if anime['mal_id'] in rated_anime]
    print(len(new_anime))
    animes = new_anime
    index_anime = {}
    anime_index = {}
    mal_index_anime = {}
    count = 0
    for anime in animes:
        anime_index[anime['title']] = count
        index_anime[count] = anime['title']
        mal_index_anime[anime['mal_id']] = count
        count += 1

    count = 0
    for user in users:
        user_index[user['username']] = count
        index_user[count] = user['username']
        wl = user['watchlist']
        count += 1
        for i in range(len(wl)):
            entry = wl[i]
            if entry[0] not in mal_index_anime:
                continue
            anime_id = mal_index_anime[entry[0]]
            user['watchlist'][i] = (anime_id, int(entry[1]))
            if int(entry[1]) != 0:
                triplets.append((anime_id, user_index[user['username']], int(entry[1])))

    triplets_set = set(triplets)
    print('Generated ' + str(len(triplets_set)) + 'triplets.')

    # Get pairs
    # cur_id = 0
    # for anime in animes:
    #     for genre in anime['genres']:
    #         if genre != "" and genre != "None":
    #             pairs.append((cur_id, tags[genre]))
    #     for tag in anime['tags']:
    #         if tag != "" and tag != "None":
    #             pairs.append((cur_id, tags[tag]))
    #     cur_id += 1
    # pairs_set = set(pairs)


def save_to_file(object, path):
    with open(path, 'wb') as f:
        pickle.dump(object, f, pickle.HIGHEST_PROTOCOL)


def load_from_file(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_all():
    global animes
    global triplets
    global triplets_set
    global anime_index
    global index_anime
    global index_user
    global user_index
    global rated_anime
    global users
    save_to_file(animes, '../../resources/user_based/animes.pkl')
    save_to_file(triplets, '../../resources/user_based/triplets.pkl')
    save_to_file(anime_index, '../../resources/user_based/anime_index.pkl')
    save_to_file(index_anime, '../../resources/user_based/index_anime.pkl')
    save_to_file(index_user, '../../resources/user_based/index_user.pkl')
    save_to_file(user_index, '../../resources/user_based/user_index.pkl')
    save_to_file(rated_anime, '../../resources/user_based/rated_anime.pkl')
    save_to_file(users, '../../resources/user_based/users.pkl')


def load_all():
    global animes
    global triplets
    global triplets_set
    global anime_index
    global index_anime
    global index_user
    global user_index
    global rated_anime
    global users
    animes = load_from_file('../../resources/user_based/animes.pkl')
    triplets = load_from_file('../../resources/user_based/triplets.pkl')
    triplets_set = set(triplets)
    anime_index = load_from_file('../../resources/user_based/anime_index.pkl')
    index_anime = load_from_file('../../resources/user_based/index_anime.pkl')
    index_user = load_from_file('../../resources/user_based/index_user.pkl')
    user_index = load_from_file('../../resources/user_based/user_index.pkl')
    rated_anime = load_from_file('../../resources/user_based/rated_anime.pkl')
    users = load_from_file('../../resources/user_based/users.pkl')


def generate_batch(triplets, n=100, negative_ratio=1.0, classification = False):
    """Generate batches of samples for training.
       Random select positive samples
       from pairs and randomly select negatives."""
    global animes
    global triplets_set
    # global neg_label

    # Create empty array to hold batch
    batch_size = n
    batch = np.zeros((int(batch_size), 3))

    # # Adjust label based on task
    # if classification:
    #     neg_label = 0
    # else:
    #     neg_label = -1

    # Continue to yield samples
    while True:
        # Randomly choose positive examples
        for idx, (anime_id, user_id, rating) in enumerate(random.sample(triplets, n)):
            batch[idx, :] = (anime_id, user_id, rating/10)
        idx += 1

        # # Add negative examples until reach batch size
        # while idx < batch_size:
        #
        #     # Random selection
        #     random_anime = random.randrange(len(animes))
        #     random_user = random.randrange(len(users))
        #
        #     # Check to make sure this is not a positive example
        #     if (random_anime, random_user) not in pairs_set:
        #         # Add to batch and increment index
        #         batch[idx, :] = (random_anime, random_user, neg_label)
        #         idx += 1

        # Make sure to shuffle order
        np.random.shuffle(batch)
        yield {'anime': batch[:, 0], 'tag': batch[:, 1]}, batch[:, 2]


def extract_weights(name, model):
    """Extract weights from a neural network model"""

    # Extract weights
    weight_layer = model.get_layer(name)
    weights = weight_layer.get_weights()[0]

    # Normalize
    weights = weights / np.linalg.norm(weights, axis=1).reshape((-1, 1))
    return weights


# init()
# save_all()
load_all()
model = anime_embedding_model(anime_index, user_index)
model.summary()
model.load_weights('models/user_based_new.h5')

# n_positive = 1024
#
# gen = generate_batch(triplets, n_positive, negative_ratio=2)
#
# # Train
# h = model.fit_generator(gen, epochs=15,
#                         steps_per_epoch=len(triplets) // n_positive,
#                         verbose=2)
# model.save('models/user_based_new.h5')

# Extract embeddings
anime_weights = extract_weights('anime_embedding', model)
tag_weights = extract_weights('tag_embedding', model)
find_similar('To Love-Ru', anime_weights, anime_index, index_anime, user_index, index_user, n=20)
