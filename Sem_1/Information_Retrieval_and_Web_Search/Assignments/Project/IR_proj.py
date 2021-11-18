# Name: Sai Anish Garapati
# UIN: 650208577

import requests
from bs4 import BeautifulSoup
from collections import deque

import os, string, nltk, re, math
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')



class crawler:
    def __init__(self):
        pass


class preprocessing:
    def __init__(self):
        pass


class vector_space_model:
    def __init__(self):
        pass


class user_query:
    def __init__(self):
        pass


def crawler(url, url_queue, page_content):
    page_content.append(BeautifulSoup(requests.get(url, verify=False).text, features='lxml'))

    for link in page_content[-1].findAll('a'):
        href = link.get('href')
        title = link.string
        if href not in url_queue:
            url_queue.append(href)
        
    return url_queue, page_content

if __name__ == '__main__':
    base_url = 'https://cs.uic.edu/'
    url_queue = deque([])
    page_content = []

    url_queue, page_content = crawler(base_url, url_queue, page_content)
    
    print(url_queue)
    print(page_content)

