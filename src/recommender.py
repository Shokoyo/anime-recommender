import numpy as np
from enum import Enum
import pickle
import requests
import requests_cache
import inspect
from triarray import TriMatrix


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


requests_cache.install_cache(cache_name='recommender_cache', backend='sqlite', expire_after=600)
quadratic_weights = (0, -4, -2, -1, 0, 0.5, 1, 2, 4, 8, 16)
default_types = (Type.TV, Type.TV_SHORT, Type.MOVIE, Type.SPECIAL, Type.OVA, Type.ONA, Type.MUSIC)
tri = np.load('../resources/tag_based/trimatrix.npy')
matrix = TriMatrix(tri, diag_val=1.0)
PROXER_API_URL = 'https://proxer.me/api/v1/user/list'
del tri

with open('../resources/proxer_api_key', 'r') as f:
    proxer_api_key = f.readlines()[0].strip()
PROXER_HEADER = {'proxer-api-key': proxer_api_key}
with open('../resources/tag_based/mal_index.pkl', 'rb') as f:
    mal_index = pickle.load(f)
with open('../resources/tag_based/anilist_index.pkl', 'rb') as f:
    anilist_index = pickle.load(f)
with open('../resources/tag_based/animes.pkl', 'rb') as f:
    animes = pickle.load(f)
with open('../resources/tag_based/proxer_index.pkl', 'rb') as f:
    proxer_index = pickle.load(f)


def recommend_anime(user, n=30, types=default_types, weighting_type=WeightingType.ABOVE_AVG_LINEAR, site=Site.ANY,
                    genres=(), tags=()):
    if n is None:
        n = 30
    if types is None:
        types = default_types
    if weighting_type is None:
        weighting_type = WeightingType.ABOVE_AVG_LINEAR
    if site is None:
        site = Site.ANY
    if genres is None:
        genres = []
    if tags is None:
        tags = []
    types = [Type(t) for t in types]
    weighting_type = WeightingType(weighting_type)
    site = Site(site)
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    print('function name "%s"' % inspect.getframeinfo(frame)[2])
    for i in args:
        print("    %s = %s" % (i, values[i]))
    if site == Site.ANY:
        # try all sites, take the first that yields a result
        recommendations = recommend_anime_mal(user, n, types, weighting_type, genres, tags)
        if recommendations:
            return recommendations
        recommendations = recommend_anime_proxer(user, n, types, weighting_type, genres, tags)
        if recommendations:
            return recommendations
    elif site == Site.MAL:
        return recommend_anime_mal(user, n, types, weighting_type, genres, tags)
    elif site == Site.PROXER:
        return recommend_anime_proxer(user, n, types, weighting_type, genres, tags)


def recommend_anime_mal(user, n=50, types=default_types, weighting_type=WeightingType.QUADRATIC_PERSONAL,
                        genres=(), tags=()):
    # get watchlist and scores
    watchlist, scores = get_mal_watchlist(user)
    return recommend_anime_generic(scores, watchlist, n, types, weighting_type, genres, tags)


def recommend_anime_proxer(user, n=50, types=default_types, weighting_type=WeightingType.QUADRATIC_PERSONAL,
                        genres=(), tags=()):
    watchlist, scores = get_proxer_watchlist(user)
    print(watchlist)
    print(scores)
    return recommend_anime_generic(scores, watchlist, n, types, weighting_type, genres, tags)


def recommend_anime_generic(scores, watchlist, n, types, weighting_type, genres, tags, consider_unrated=False):
    type_names = [Type(t).name for t in types]
    avg_score = np.average(scores)
    if avg_score == 0:
        avg_score = 5
    if len(watchlist) == 0:
        return []
    scores_filtered = scores[:]
    watchlist_filtered = watchlist[:]
    # Remove unrated entries or use avg score
    if consider_unrated:
        for i in range(len(scores)):
            if scores[i] == 0:
                scores_filtered[i] = avg_score
        print(scores_filtered)
    else:
        non_zero_indices = []
        for i in range(len(scores)):
            if scores[i] != 0:
                non_zero_indices.append(i)
        scores_filtered = [scores[i] for i in non_zero_indices]
        watchlist_filtered = [watchlist[i] for i in non_zero_indices]
        if len(scores_filtered) == 0:
            return recommend_anime_generic(scores, watchlist, n, types, weighting_type, genres, tags, True)

    similarity_vec = get_similar(watchlist_filtered, scores_filtered, weighting_type)
    sorted_vec = [i for i in sorted(enumerate(similarity_vec), key=lambda x: x[1], reverse=True)]
    largest = sorted_vec[0][1]
    sorted_vec = [(i[0], i[1] / largest) for i in sorted_vec]
    recommendations = []
    cur = 0
    while len(recommendations) < n and cur < len(sorted_vec):
        if sorted_vec[cur][0] not in watchlist:
            conforms = True
            anime = get_starting_anime(animes[sorted_vec[cur][0]], watchlist, [])
            for genre in genres:
                if genre not in anime['genres']:
                    conforms = False
            for tag in tags:
                if tag not in [i[0] for i in anime['tags']]:
                    conforms = False
            if anime['type'] not in type_names:
                conforms = False
            if anilist_index[anime['anime_id']] in [r[0] for r in recommendations]:
                conforms = False
            if conforms:
                recommendations.append((anilist_index[anime['anime_id']], sorted_vec[cur][1]))
        cur += 1
    print(cur)
    max_width = max([len(animes[r[0]]['title']) for r in recommendations])
    for rec in recommendations:
        print(f'{animes[rec[0]]["title"]:{max_width + 3}} Similarity: {rec[1]:.{3}}')
    return [(int(animes[r[0]]['anime_id']), r[1]) for r in recommendations]


# get the "starting point" for an OVA or sequel anime ("earliest" anime not on watchlist)
def get_starting_anime(anime, watchlist, visited):
    visited.append(anime['anime_id'])
    for relation in anime['relations']:
        for relation_id, relation_type in relation.items():
            if relation_id not in anilist_index:
                continue
            if relation_id in visited:
                if animes[anilist_index[relation_id]]['type'] == 'TV':
                    if relation_type == 'PREQUEL' or relation_type == 'PARENT':
                        return animes[anilist_index[relation_id]]
                    else:
                        continue
                continue
            if relation_type == 'PREQUEL' and anilist_index[relation_id] not in watchlist:
                visited.append(relation_id)
                return get_starting_anime(animes[anilist_index[relation_id]], watchlist, visited)
            elif relation_type == 'PARENT' and anilist_index[relation_id] not in watchlist:
                visited.append(relation_id)
                return get_starting_anime(animes[anilist_index[relation_id]], watchlist, visited)
    return anime


def get_mal_watchlist_page(user, page):
    url = 'https://myanimelist.net/animelist/' + user + '/load.json?offset=' + str(300*(page-1)) + '&status=7'
    request = requests.get(url)
    if request.status_code != 200:
        return []
    else:
        return request.json()


def get_mal_watchlist(user):
    i = 1
    loop = True
    watchlist = []
    scores = []
    while loop:
        print(i)
        watchlist_page = get_mal_watchlist_page(user, i)
        if len(watchlist_page) == 0:
            loop = False
        for anime in watchlist_page:
            mal_id = str(anime['anime_id'])
            if mal_id not in mal_index:
                continue
            # if 'Comedy' not in animes[mal_index[mal_id]]['genres'] or 'Romance' not in animes[mal_index[mal_id]]['genres']:
            #     continue
            watchlist.append(mal_index[mal_id])
            scores.append(anime['score'])
        i += 1
    return watchlist, scores


def get_proxer_watchlist(user):
    watchlist = []
    scores = []
    i = 0
    finished = False
    while not finished:
        request = requests.get(PROXER_API_URL, params={
            'username': user,
            'kat': 'anime',
            'p': i
        }, headers=PROXER_HEADER)
        i += 1
        result = request.json()
        print(result)
        for entry in result['data']:
            if entry['state'] != 2 and int(entry['id']) in proxer_index:
                watchlist.append(proxer_index[int(entry['id'])])
                scores.append(int(entry['rating']))
        if len(result['data']) == 0 or result['error'] == 1:
            finished = True
    return watchlist, scores


def get_similar(watchlist, scores, weighting_type):
    similarity_vector = np.zeros(len(matrix[0]))
    scores = get_weights(scores, weighting_type)
    for i in range(len(watchlist)):
        similarity_vector += scores[i] * get_row(watchlist[i])
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
        print(scores)
    elif weighting_type == WeightingType.LINEAR:
        return scores
    if all(v == 0 for v in scores):
        scores = [score + 1 for score in scores]
    return scores


def get_row(i):
    global matrix
    return matrix[i]


# recommend_anime_mal('Shokoyo', weighting_type=WeightingType.ABOVE_AVG_LINEAR)