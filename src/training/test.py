import csv

path = r'E:\Benutzer\Eigene Dokumente\Waifu Laifu\Recommender\anilist-anime-new'
curGenreId = 0
genreIds = {}
# with open(path, mode='r', encoding="utf8") as csv_file:
#     csv_reader = csv.DictReader(csv_file)
#     line_count = 0
#     for row in csv_reader:
#         if line_count == 0:
#             print(f'Column names are {", ".join(row)}')
#             line_count += 1
#         genres = row['genre'].split(", ")
#         for genre in genres:
#             if genre not in genreIds and genre != "":
#                 genreIds[genre] = curGenreId
#                 curGenreId += 1
#         line_count += 1
#     print(f'Processed {line_count} lines.')
#     for genre in genreIds.keys():
#         print(genre + " : " + str(genreIds[genre]))
# #filter anime
# path = r'E:\Benutzer\Eigene Dokumente\Waifu Laifu\Recommender\lists'
with open(path + r'\anime.csv', mode='r', encoding="utf8") as csv_file:
    csv_reader = csv.DictReader(csv_file)
    with open(path + r'\anime_filtered_new.csv', mode='w', encoding="utf8", newline='') as outfile:
        csv_writer = csv.DictWriter(outfile, fieldnames=csv_reader.fieldnames, extrasaction='ignore')
        csv_writer.writeheader()
        for row in csv_reader:
            if (row['genres'] != "" and row['genres'] != "\"\"") or (row['tags'] != "" and row['tags'] != "\"\""):
                if row['popularity'] != '' and row['score'] != '':
                    csv_writer.writerow(row)
