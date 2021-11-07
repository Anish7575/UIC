Name: Sai Anish Garapati

1.a) The program submitted along, processes the Citeseer UMD corpus by tokenizing the text and grouping the words based on their frequencies. The following steps were done:
-> All the text from files is read using 'os' library from python
-> 'word_tokenize' module from nltk.tokenize is used to tokenize the text on whitespace.
-> All the punctutaions are removed based on the punctuations from the list 'string.punctuation'
-> From the resultant list of words which is also the Collection, a dictionary is created with the word as key and its frequency in the collection as value, which forms the Vocabulary.
-> Stop words are removed from the vocabulary based on the 'stopwords' module from nltk.corpus library
-> 'PorterStemmer' from nltk.stem is used to stem the words and regroup the words in the dictionary based on the stemmed words.

1.b) Instructions for running the code ?

2.a) The total number of words in collection are 477989

2.b) The vocabulary size is 19630

2.c) Top 20 words in the ranking and their respective frequencies in the collection:
{'the': 25667, 'of': 18643, 'and': 14134, 'a': 13372, 'to': 11539, 'in': 10069, 'for': 7382, 'is': 6580, 'we': 5147, 'that': 4821, 'this': 4447, 'are': 3738, 'on': 3653, 'an': 3281, 'with': 3200, 'as': 3060, 'by': 2767, 'data': 2694, 'be': 2500, 'information': 2326}

2.d) Stop words from the top 20 words:
['the', 'of', 'and', 'a', 'to', 'in', 'for', 'is', 'we', 'that', 'this', 'are', 'on', 'an', 'with', 'as', 'by', 'be']

2.e) The count of unique words accounting for 15% of the total words in collection is 0.

3.a) The total number of words in the new collection are 294927

3.b) The new vocabulary size is 13625

3.c) Top 20 words in the ranking and their respective frequencies in the new collection:
{'data': 2694, 'inform': 2402, 'paper': 2247, 'system': 3745, '1': 1569, 'agent': 2695, 'web': 1440, 'learn': 1742, 'use': 3741, 'base': 1499, 'model': 2314, 'user': 1758, 'approach': 1544, 'queri': 1905, 'problem': 1545, 'search': 1144, 'new': 976, 'introduct': 952, 'result': 1202, 'applic': 1524}

3.d) There are no stop words from the Top 20 words from new collection

3.e) The count of unique words accounting for 15% of the total words in the collection is 0.
