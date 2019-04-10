import scrapy
from bs4 import BeautifulSoup
import pickle
import time


class ProxerSpider(scrapy.Spider):
    name = "proxer"
    with open('../../resources/tag_based/animes_unfiltered.pkl', 'rb') as f:
        animes = pickle.load(f)

    index = {}
    index_native = {}
    index_english = {}

    for i in range(len(animes)):
        anime = animes[i]
        index[anime['title'].lower()] = i

    def start_requests(self):
        urls = [
            'https://proxer.me/anime/animeseries/rating/all/' + str(i + 1) for i in range(106)
        ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        # soup = BeautifulSoup(response.text)
        # for entry in soup.findAll("a", {"class": "tip"}):
        #    yield {'title': entry.text, 'id': entry['href'].split('/')[-1].strip('#top')}
        for link in response.css('a.tip::attr(href)').getall():
            yield response.follow(link, callback=self.parse_anime)

    def parse_anime(self, response):
        soup = BeautifulSoup(response.text, 'lxml')
        synonyms = []
        error = False
        original_title = soup(text='Original Titel')
        if len(original_title) > 0:
            original_title = original_title[0].parent.find_next('td').text
        else:
            error = True
        japanese_title = soup(text='Japanischer Titel')
        if len(japanese_title) > 0:
            japanese_title = japanese_title[0].parent.find_next('td').text
        synonym_elems = soup(text='Synonym')
        for elem in synonym_elems:
            synonyms.append(elem.parent.find_next('td').text)

        if not error:
            yield {
                'original_title': original_title,
                'japanese_title': japanese_title,
                'synonyms': synonyms,
            }
        else:
            yield {
                'original_title': response.request.url,
                'japanese_title': 'error',
                'synonyms': [],
            }
