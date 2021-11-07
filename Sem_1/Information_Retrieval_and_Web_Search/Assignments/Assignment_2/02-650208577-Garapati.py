# Name: Sai Anish Garapati
# UIN: 650208577

import os, string, nltk, re, math
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# nltk.download('punkt')
# nltk.download('stopwords')

# Remove stopwords from documents and queries
def remove_stop_words(words_list):
    stop_words = set(stopwords.words('english'))
    return [word for word in words_list if word not in stop_words]

# Apply stemmer on the documents and queries
def stemmer(words_list):
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    return [ps.stem(word) for word in words_list if ps.stem(word) not in stop_words]

# Preprocessing individual document and query
def word_preprocessing(content):
    content = re.sub(re.compile('<.*>\n'), '', content)
    content = re.sub(re.compile('[0-9]'), '', content)

    words_list = word_tokenize(content);
    words_list = [word.translate(str.maketrans(
            '', '', string.punctuation)) for word in words_list]
    words_list = [word.lower() for word in words_list if word != '']

    words_list = remove_stop_words(words_list)
    words_list = stemmer(words_list)

    return [word for word in words_list if len(word) > 2]

# Processing the document corpus
def preprocessing(path):
    file_list = os.listdir(path)
    docs_list = []
    for file_name in file_list:
        file = open(path + file_name, 'r')
        content = file.read()

        docs_list.append(word_preprocessing(content))

    return docs_list

# Building the inverted index table
def build_inverted_index(docs):
    inverted_index = {}
    for i in range(0, len(docs)):
        for word in docs[i]:
            if word in inverted_index:
                if i not in inverted_index[word]:
                    inverted_index[word].update({i: 0})
            else:
                inverted_index[word] = {i: 0}
            inverted_index[word][i] += 1
    return inverted_index

# Computing the document vector lengths
def compute_docs_length(index, docs_list):
    docs_length = [0.0] * len(docs_list)

    for i in range(0, len(docs_list)):
        for word in docs_list[i]:
            # Using length of dictionary corresponding to a word as document frequency
            docs_length[i] += (index[word][i] * math.log(float(len(docs_list)/len(index[word]))))**2
        docs_length[i] = math.sqrt(docs_length[i])
    return docs_length

# Processing the queries
def query_preprocessing(path):
    file = open(path, 'r')
    content = file.read().split('\n')
    queries_list = []
    for query in content:
        query_dict = {}
        query_list = word_preprocessing(query)
        for item in query_list:
            if item in query_dict:
                query_dict[item] += 1
            else:
                query_dict[item] = 1
        queries_list.append(query_dict)

    return queries_list

def compute_ranked_docs(index, docs_length, queries):
    ranked_docs_all = []
    for i in range(0, len(queries)):
        ranked_docs_query = [0.0] * len(docs_length)
        query_length = 0.0
        for query_term in queries[i]:
            if query_term in index:
                for key in index[query_term]:
                    ranked_docs_query[key] += (index[query_term][key] * queries[i][query_term]) * (math.log(float(len(docs_length))/float(len(index[query_term]))))**2
                query_length += (queries[i][query_term] * math.log(float(len(docs_length))/float(len(index[query_term]))))**2
        for j in range(0, len(docs_length)):
            ranked_docs_query[j] /= (docs_length[j] * math.sqrt(query_length))
        ranked_docs_all.append([(j + 1, ranked_docs_query[j]) for j in range(0, len(docs_length))])
        ranked_docs_all[-1] = sorted(ranked_docs_all[-1], key = lambda x: x[1], reverse = True)
        ranked_docs_all[-1] = [(i + 1, rank[0]) for rank in ranked_docs_all[-1]]

    return ranked_docs_all

if __name__ == '__main__':
    corpus_path = input('Enter document corpus directory path\n') + '/'
    queries_path = input('Enter queries file path\n')
    relevance_path = input('Enter relevance queries file path\n')

    docs_list = preprocessing(corpus_path)

    inverted_index = build_inverted_index(docs_list)
    print('Computed inverted index table from document corpus: ', inverted_index, '\n')

    docs_length = compute_docs_length(inverted_index, docs_list)
    print('Computed document vector lengths: ', docs_length, '\n')

    queries_list = query_preprocessing(queries_path)
    print('Queries after processing: ', queries_list, '\n')

    ranked_docs_all = compute_ranked_docs(inverted_index, docs_length, queries_list)
    print('retreived ranked documents: ', ranked_docs_all, '\n')

    relevance_queries = [[] for _ in range(len(queries_list))]
    file = open(relevance_path, 'r')
    for line in file.read().split('\n'):
        q, r = line.split(' ')
        relevance_queries[int(q) - 1].append(int(r))

    for n in [10, 50, 100, 500]:
        avg_precision = avg_recall = 0.0
        for i in range(0, len(relevance_queries)):
            results = ranked_docs_all[i][:n]
            tp = fp = fn = 0
            for rel_doc in relevance_queries[i]:
                if rel_doc in [res[1] for res in results]:
                    tp += 1
                else:
                    fn += 1
            fp = len(results) - tp
            avg_precision += (tp/(tp + fp))
            avg_recall += (tp/(tp + fn))
            print('Precision for query {} for top {} documents in the ranking: '.format(i + 1, n), tp/(tp + fp))
            print('Recall for query {} for top {} documents in the ranking: '.format(i + 1, n), tp/(tp + fn), '\n')

        avg_precision /= len(relevance_queries)
        avg_recall /= len(relevance_queries)

        print('Average precision over {} queries for top {} documents in ranking: '.format(len(relevance_queries), n), avg_precision)
        print('Average recall over {} queries for top {} documents in ranking: '.format(len(relevance_queries), n), avg_recall)

        print('-----------------------------------------------------------------------\n')
