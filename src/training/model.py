from keras.layers import Input, Embedding, Dot, Reshape, Dense
from keras.models import Model


def anime_embedding_model(anime_index, tag_index, embedding_size=15, classification=False):
    """Model to embed books and wikilinks using the functional API.
       Trained to discern if a tag is present in a article"""

    # Both inputs are 1-dimensional
    anime = Input(name='anime', shape=[1])
    tag = Input(name='tag', shape=[1])

    # Embedding the anime (shape will be (None, 1, 50))
    anime_embedding = Embedding(name='anime_embedding',
                               input_dim=len(anime_index),
                               output_dim=embedding_size)(anime)

    # Embedding the tag (shape will be (None, 1, 50))
    tag_embedding = Embedding(name='tag_embedding',
                               input_dim=len(tag_index),
                               output_dim=embedding_size)(tag)

    # Layer for popularity estimation
    popularity_output = Dense(1)(anime_embedding)
    popularity_output = Reshape(target_shape=[1], name='popularity_output')(popularity_output)

    # Layer for rating estimation
    score_output = Dense(1)(anime_embedding)
    score_output = Reshape(target_shape=[1], name='score_output')(score_output)

    # Merge the layers with a dot product along the second axis (shape will be (None, 1, 1))
    merged = Dot(name='dot_product', normalize=True, axes=2)([anime_embedding, tag_embedding])

    # Reshape to be a single number (shape will be (None, 1))
    merged = Reshape(target_shape=[1], name='tag_output')(merged)

    # If classifcation, add extra layer and loss function is binary cross entropy
    if classification:
        merged = Dense(1, activation='sigmoid')(merged)
        model = Model(inputs=[anime, tag], outputs=[merged, score_output, popularity_output])
        model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Otherwise loss function is mean squared error
    else:
        model = Model(inputs=[anime, tag], outputs=[merged, score_output, popularity_output])
        model.compile(optimizer='Adam', loss='mse', loss_weights=[0.65, 0.2, 0.15])

    return model
