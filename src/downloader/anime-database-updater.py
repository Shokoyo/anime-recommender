from google.cloud import firestore
import google.cloud
import requests
import time

db = firestore.Client()
col_ref = db.collection('anime')

query = '''
query ($page: Int) {
Page (page: $page, perPage: 50) {
        pageInfo {
            total
            currentPage
            lastPage
            hasNextPage
            perPage
        }
        media (type: ANIME) {
            id
            idMal
            format
            meanScore
            popularity
            status
            source
            genres
            coverImage {
                large
                medium
            }
            bannerImage
            tags {
                id
                name
                rank
            }
            title {
                english
                native
                romaji
            }
            relations {
                edges {
                    node {
                        id
                        title {
                            romaji
                        }
                    }
                    relationType
                }
            }
        }
    }
}
'''

url = 'https://graphql.anilist.co'

# for i in range(1, 254):
#     response = requests.post(url, json={'query': query, 'variables': {'page': i}})
#     if response.status_code != 200:
#         print(response.status_code)
#         print(response.json())
#     for e in response.json()['data']['Page']['media']:
#         new_relations = []
#         for relation in e['relations']['edges']:
#             new_relations.append({str(relation['node']['id']): relation['relationType']})
#         e['relations'] = new_relations
#         col_ref.document(str(e['id'])).set(e)
#     time.sleep(0.7)
