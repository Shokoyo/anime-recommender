#!flask/bin/python
from flask import Flask, request, jsonify
from flask_cors import CORS
import recommender

__name__ = '__main__'
app = Flask(__name__)
CORS(app)
error = {'error':
                    {
                        'id': 404,
                        'message': 'User not found or watchlist empty'
                    }
                 }

def handle_list(list):
    if list is None:
        return jsonify(error)
    elif len(list) == 0:
        return jsonify(error)
    else:
        return jsonify({'data': list})


@app.route('/api/v1.0/recommend/<string:user>', methods=['GET', 'POST'])
def get_recommendations(user):
    print(request)
    body = request.get_json()
    if body is None:
        response = recommender.recommend_anime(user)
        return handle_list(response)
    else:
        return handle_list(recommender.recommend_anime(
            user,
            n=body.get('n'),
            types=body.get('types'),
            genres=body.get('genres'),
            tags=body.get('tags'),
            weighting_type=body.get('weighting_type'),
            site=recommender.Site(body.get('site'))
          )
        )


if __name__ == '__main__':
    app.run(debug=False)
