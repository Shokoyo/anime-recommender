import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# from IPython import get_ipython
# get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('fivethirtyeight')
plt.rcParams['font.size'] = 15


def find_similar(name, weights, anime_index, index_anime, tag_index, index_tag, index_name='anime', n=10, least=False, return_dist=False, plot=False):
    """Find n most similar items (or least) to name based on embeddings. Option to also plot the results"""

    # Select index and reverse index
    if index_name == 'anime':
        index = anime_index
        rindex = index_anime
    elif index_name == 'tag':
        index = tag_index
        rindex = index_tag

    # Check to make sure `name` is in index
    try:
        # Calculate dot product between anime and all others
        dists = np.dot(weights, weights[index[name]])
    except KeyError:
        print(f'{name} Not Found.')
        return

    # Sort distance indexes from smallest to largest
    sorted_dists = np.argsort(dists)

    # Plot results if specified
    if plot:

        # Find furthest and closest items
        furthest = sorted_dists[:(n // 2)]
        closest = sorted_dists[-n - 1: len(dists) - 1]
        items = [rindex[c] for c in furthest]
        items.extend(rindex[c] for c in closest)

        # Find furthest and closets distances
        distances = [dists[c] for c in furthest]
        distances.extend(dists[c] for c in closest)

        colors = ['r' for _ in range(n // 2)]
        colors.extend('g' for _ in range(n))

        data = pd.DataFrame({'distance': distances}, index=items)

        # Horizontal bar chart
        data['distance'].plot.barh(color=colors, figsize=(10, 8),
                                   edgecolor='k', linewidth=2)
        plt.xlabel('Cosine Similarity');
        plt.axvline(x=0, color='k');

        # Formatting for italicized title
        name_str = f'{index_name.capitalize()}s Most and Least Similar to'
        for word in name.split():
            # Title uses latex for italize
            name_str += ' $\it{' + word + '}$'
        plt.title(name_str, x=0.2, size=28, y=1.05)

        return None

    # If specified, find the least similar
    if least:
        # Take the first n from sorted distances
        closest = sorted_dists[:n]

        print(f'{index_name.capitalize()}s furthest from {name}.\n')

    # Otherwise find the most similar
    else:
        # Take the last n sorted distances
        closest = sorted_dists[-n:]

        # Need distances later on
        if return_dist:
            return dists, closest

        print(f'{index_name.capitalize()}s closest to {name}.\n')

    # Need distances later on
    if return_dist:
        return dists, closest

    # Print formatting
    max_width = max([len(rindex[c]) for c in closest])

    # Print the most similar and distances
    for c in reversed(closest):
        print(f'{index_name.capitalize()}: {rindex[c]:{max_width + 2}} Similarity: {dists[c]:.{2}}')
