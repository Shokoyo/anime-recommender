#!flask/bin/python
from flask import Flask, request, jsonify
import recommender

__name__ = '__main__'
app = Flask(__name__)


def handle_list(list):
    if len(list) == 0:
        error = {'error':
                    {
                        'id': 404,
                        'message': 'User not found or watchlist empty'
                    }
        }
        return jsonify(error)
    else:
        return jsonify({'data': list})


@app.route('/api/v1.0/recommend/<string:user>', methods=['GET'])
def get_recommendations(user):
    print(request)
    body = request.get_json()
    if body is None:
        response = recommender.recommend_anime(user)
        return handle_list(response)
        return jsonify(response)
    else:
        return jsonify(recommender.recommend_anime(
            user,
            n=body.get('n'),
            types=body.get('types'),
            genres=body.get('genres'),
            weighting_type=body.get('weighting_type'),
            site=body.get('site')
          )
        )


if __name__ == '__main__':
    app.run(debug=False)
