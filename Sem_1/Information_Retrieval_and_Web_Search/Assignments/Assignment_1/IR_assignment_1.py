import os
import string
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


def preprocessing(path):
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
	return words_freq_stemmed


path = 'citeseer/'
words_list = preprocessing(path)

word_freq = list_to_word_freq(words_list)

print(len(words_list))

print(len(word_freq))

word_freq = dict(
	sorted(word_freq.items(), key=lambda item: item[1], reverse=True))
word_freq_top_20 = dict(list(word_freq.items())[:20])
print(word_freq_top_20)

stop_words = set(stopwords.words('english'))
print([word for word in word_freq_top_20 if word in stop_words])

print(len([word for word in word_freq if word_freq[word] >= 0.15 * len(words_list)]))

# Removing stop words from the vocabulary
words_freq_new = remove_stop_words_from_dict(word_freq)
# Applying stemmer on the new vocabulary
words_freq_new_stemmed = stemmer_on_dict(words_freq_new)

print(sum(words_freq_new_stemmed.values()))

print(len(words_freq_new_stemmed))

print(dict(list(words_freq_new_stemmed.items())[:20]))

print([word for word in dict(
	list(words_freq_new_stemmed.items())[:20]) if word in stop_words])

print(len([word for word in words_freq_new_stemmed if words_freq_new_stemmed[word]
           >= 0.15 * sum(words_freq_new_stemmed.values())]))
