# Name: Sai Anish Garapati
# UIN: 650208577

# Importing required Libraries

import os, string, nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# nltk.download('punkt')
# nltk.download('stopwords')

# Defining preprocessing and utility functions

def word_preprocessing(path):
	file_list = os.listdir(path)

	file_string = ''
	for file_name in file_list:
		file = open(path + file_name, 'r')
		file_string += file.read()

	words_list = word_tokenize(file_string)
	words_list = [word.translate(str.maketrans(
		'', '', string.punctuation)) for word in words_list]
	words_list = [word.lower() for word in words_list if word != '']
	return words_list


def list_to_word_freq(words_list):
	word_freq = {}
	for word in words_list:
		if word in word_freq:
			word_freq[word] += 1
		else:
			word_freq[word] = 1
	return word_freq


def remove_stop_words_from_dict(words_freq):
	return {word: freq for (word, freq) in word_freq.items() if word not in stop_words}


def stemmer_on_dict(words_freq):
	ps = PorterStemmer()
	words_freq_stemmed = {}

	for word, freq in words_freq.items():
		word = ps.stem(word)
		if (word in words_freq_stemmed):
			words_freq_stemmed[word] += freq
		else:
			words_freq_stemmed[word] = freq
	return dict(sorted(words_freq_stemmed.items(), key=lambda item: item[1], reverse=True))

if (__name__ == '__main__'):

	# Preprocessing the collection
	path = 'citeseer/'
	words_list = word_preprocessing(path)

	# Converting list into dictionary with <word, frequency> as <key, value> pair 
	word_freq = list_to_word_freq(words_list)

	print('')

	# 2.a) Total number of words in the collection
	print('2.a) Total number of words in the collection:', len(words_list), '\n')

	# 2.b) Vocabulary size
	print('2.b) Vocabulary size:', len(word_freq), '\n')

	# 2.c) Top 20 words in the ranking
	# Sorting dictionary in decreasing order based on frequencies
	word_freq = dict(
		sorted(word_freq.items(), key=lambda item: item[1], reverse=True))
	print('2.c) Top 20 words in ranking from the vocabulary:\n', dict(list(word_freq.items())[:20]), '\n')

	# 2.d) Stop words from the top 20 words in ranking
	stop_words = set(stopwords.words('english'))
	print('2.d) Stop words from the top 20 words in ranking:\n', [
        word for word in dict(list(word_freq.items())[:20]) if word in stop_words], '\n')

	# 2.e) Count of unique words accounting for 15% of the total words in the collection
	print('2.e) Count of Unique words accounting for 15% of the total words in the collection:',
			len([word for word in word_freq if word_freq[word] >= 0.15 * len(words_list)]), '\n')
	
	# 3) Integrating stemmer and stop word eliminator
	# Removing stop words from the vocabulary
	words_freq_new = remove_stop_words_from_dict(word_freq)

	# Applying stemmer on the new vocabulary
	words_freq_new_stemmed = stemmer_on_dict(words_freq_new)

	# 3.a) Total number of words in the new collection
	print('3.a) Total number of words in the new collection:', sum(words_freq_new_stemmed.values()), '\n')

	# 3.b) New Vocabulary size
	print('3.b) New Vocabulary size:', len(words_freq_new_stemmed), '\n')

	# 3.c) Top 20 words in the ranking
	print('3.c) Top 20 words in the ranking:\n', dict(list(words_freq_new_stemmed.items())[:20]), '\n')

	# 3.d) Stop words from the top 20 in ranking
	print('3.d) Stop words from the top 20 in ranking:\n', [word for word in dict(
		list(words_freq_new_stemmed.items())[:20]) if word in stop_words], '\n')
	
	# 3.e) Count of unique words accounting for 15% of the total words in the new collection
	print('3.e) Count of Unique words accounting for 15% of the total words in the new collection:',
       len([word for word in words_freq_new_stemmed if words_freq_new_stemmed[word] >= 0.15 * sum(words_freq_new_stemmed.values())]), '\n')
