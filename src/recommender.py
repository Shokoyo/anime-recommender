import numpy as np
from enum import Enum
from jikanpy import Jikan
import pickle
import time
import jikanpy
import requests
from triarray import TriMatrix
import triarray

class Type(Enum):
    TV = 1
    TV_SHORT = 2
    MOVIE = 3
    SPECIAL = 4
    OVA = 5
    ONA = 6
    MUSIC = 7


class Site(Enum):
    ANY = 1
    MAL = 2
    ANILIST = 3
    PROXER = 4


class WeightingType(Enum):
    LINEAR = 1
    ABOVE_AVG_LINEAR = 2
    ABOVE_AVG_OFFSET = 3
    ABOVE_AVG_QUADRATIC = 4
    QUADRATIC_FIXED = 5
    QUADRATIC_PERSONAL = 6


jikan = Jikan()
quadratic_weights = (0, -4, -2, -1, 0, 0.5, 1, 2, 4, 8, 16)
all_types = (Type.TV, Type. TV_SHORT, Type.MOVIE, Type.SPECIAL, Type.OVA, Type.ONA, Type.MUSIC)
tri = np.load('../resources/tag_based/trimatrix.npy')
matrix = TriMatrix(tri, diag_val=1.0)
del tri

with open('../resources/tag_based/mal_index.pkl', 'rb') as f:
    mal_index = pickle.load(f)
with open('../resources/tag_based/anilist_index.pkl', 'rb') as f:
    anilist_index = pickle.load(f)
with open('../resources/tag_based/animes.pkl', 'rb') as f:
    animes = pickle.load(f)


def recommend_anime(user, n=50, types=all_types, weighting_type=WeightingType.ABOVE_AVG_OFFSET, site=Site.ANY, genres=[]):
    if site == Site.ANY:
        # try all sites, take the first that yields a result
        return recommend_anime_mal(user, n, types, weighting_type)
    elif site == Site.MAL:
        return recommend_anime_mal(user, n, types, weighting_type)


def recommend_anime_mal(user, n=50, types=all_types, weighting_type=WeightingType.QUADRATIC_PERSONAL):
    # get watchlist and scores
    watchlist, scores = get_mal_watchlist(user)
    return recommend_anime_generic(scores, watchlist, n, types, weighting_type)


def recommend_anime_generic(scores, watchlist, n, types, weighting_type):
    if len(watchlist) == 0:
        return []
    similarity_vec = get_similar(watchlist, scores, weighting_type)
    sorted_vec = [i for i in sorted(enumerate(similarity_vec), key=lambda x: x[1], reverse=True)]
    largest = sorted_vec[0][1]
    sorted_vec = [(i[0], i[1] / largest) for i in sorted_vec]
    recommendations = []
    cur = 0
    start = time.time()
    while len(recommendations) < n and cur < len(sorted_vec):
        if sorted_vec[cur][0] not in watchlist:
            recommendations.append(sorted_vec[cur])
        cur += 1
    print('Recommendations: ' + str(time.time() - start))
    max_width = max([len(animes[r[0]]['title']) for r in recommendations])
    for rec in recommendations:
        print(f'{animes[rec[0]]["title"]:{max_width + 3}} Similarity: {rec[1]:.{3}}')
    return [(int(animes[r[0]]['anime_id']), r[1]) for r in recommendations]


def get_mal_watchlist_page(user, page):
    url = 'https://myanimelist.net/animelist/' + user + '/load.json?offset=' + str(300*(page-1)) + '&status=7'
    request = requests.get(url)
    return request.json()


def get_mal_watchlist(user):
    i = 1
    loop = True
    watchlist = []
    scores = []
    while loop:
        print(i)
        try:
            watchlist_page = get_mal_watchlist_page(user, i)
        except jikanpy.exceptions.APIException:
            print('API Exception occurred')
            break
        if len(watchlist_page) == 0:
            loop = False
        for anime in watchlist_page:
            mal_id = str(anime['anime_id'])
            if mal_id not in mal_index:
                continue
            if anime['score'] == 0:
                continue
            # if 'Comedy' not in animes[mal_index[mal_id]]['genres'] or 'Romance' not in animes[mal_index[mal_id]]['genres']:
            #     continue
            watchlist.append(mal_index[mal_id])
            scores.append(anime['score'])
        i += 1
    return watchlist, scores


def get_similar(watchlist, scores, weighting_type):
    similarity_vector = np.zeros(len(matrix[0]))
    score_sum = 0
    scores = get_weights(scores, weighting_type)
    for i in range(len(watchlist)):
        similarity_vector += scores[i] * get_row(watchlist[i])
        score_sum += scores[i]
    similarity_vector *= 1/score_sum
    return similarity_vector


def get_weights(scores, weighting_type):
    if weighting_type == WeightingType.QUADRATIC_FIXED:
        scores = [quadratic_weights[score] for score in scores]
    elif weighting_type == WeightingType.QUADRATIC_PERSONAL:
        avg = int(np.floor(np.average(scores)))
        weightings = np.zeros(11)
        if avg > 0:
            weightings[avg - 1] = 0.5
        for i in range(avg, 11):
            weightings[i] = pow(2, i - avg)
        for i in reversed(range(1, avg - 2)):
            weightings[i] = -1*pow(1.5, avg - i) + 1.5
        print(weightings)
    elif weighting_type == WeightingType.ABOVE_AVG_LINEAR:
        avg = np.average(scores)
        scores = [score - avg for score in scores]
    elif weighting_type == WeightingType.ABOVE_AVG_QUADRATIC:
        # scores near the average influence the result less
        avg = np.floor(np.average(scores))
        scale = (10-avg)*1.5
        scale_neg = avg - min(scores)
        print(avg)
        print(min(scores))
        for i in range(len(scores)):
            if scores[i] >= avg + 1:
                scores[i] = 1 + (12 * pow((scores[i] - avg)/scale, 2))
            else:
                scores[i] = 1 - 12*pow((avg - scores[i] - 1)/scale_neg, 2)
        print(np.sort(scores))
    elif weighting_type == WeightingType.ABOVE_AVG_OFFSET:
        avg = np.average(scores)
        scores = [score - avg + 1 for score in scores]
    elif weighting_type == WeightingType.LINEAR:
        return scores
    return scores


def get_row(i):
    global matrix
    return matrix[i]


# recommend_anime_mal('Shokoyo', weighting_type=WeightingType.ABOVE_AVG_LINEAR)
