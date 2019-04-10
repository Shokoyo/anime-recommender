import csv
import pickle

with open('../../resources/tag_based/animes_unfiltered.pkl', 'rb') as f:
    animes = pickle.load(f)

index = {}
index_native = {}
index_english = {}

error_count = 0

for i in range(len(animes)):
    anime = animes[i]
    title = anime['title'].lower().replace(' ', '').replace(':', '').replace('-', '').replace('★', '')
    if title != 'None' and title in index and index[title] != i:
        print('Title error native:' + title)
        error_count += 1
    elif title != 'none':
        index[title] = i
    title = anime['title_english'].lower().replace(' ', '').replace(':', '').replace('-', '').replace('★', '')
    if title != 'None' and title in index_english and index_english[title] != i:
        print('Title error english:' + title)
        error_count += 1
    elif title != 'none':
        index_english[title] = i

print(error_count)

with open('missing.txt', 'w', encoding='utf-8') as errorfile:
    with open('proxer_mapping.csv', 'w', encoding='utf-8') as outfile:
        csv_out = csv.DictWriter(outfile, fieldnames=['title', 'proxer_id', 'anilist_id', 'unfiltered_index'], lineterminator='\n')
        csv_out.writeheader()
        with open('proxer.csv', 'r', encoding='utf-8') as infile:
            csv_in = csv.DictReader(infile)
            for row in csv_in:
                proxer_title = row['title'].lower().replace(' ', '').replace(':', '').replace('-', '').replace('★', '')
                if proxer_title in index:
                    csv_out.writerow({
                        'title': animes[index[proxer_title]]['title'],
                        'proxer_id': row['id'],
                        'anilist_id': animes[index[proxer_title]]['anime_id'],
                        'unfiltered_index': index[proxer_title]
                    })
                elif proxer_title in index_english:
                    csv_out.writerow({
                        'title': animes[index_english[proxer_title]]['title'],
                        'proxer_id': row['id'],
                        'anilist_id': animes[index_english[proxer_title]]['anime_id'],
                        'unfiltered_index': index_english[proxer_title]
                    })
                else:
                    errorfile.write(row['title'] + ',' + str(row['id']) + '\n')
