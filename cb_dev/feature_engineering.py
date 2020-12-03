import emoji
import metapy
from nltk import ngrams
import os
import re
import numpy as np
import pandas as pd
from jx_dev.utils_text_clf import utils_text_clf as utils
import cb_dev.vocab_helpers as helper
from collections import Counter


import warnings
warnings.filterwarnings("ignore")

with open("stopwords.txt", 'r') as file:
    text = file.read()
    file.close()
    stopword_list = text.split("\n")

def extract_sequences(tok):
    sequences = []
    for token in tok:
        if token == '<s>':
            sequences.append(metapy.sequence.Sequence())
        elif token != '</s>':
            sequences[-1].add_symbol(token)
    return sequences

def run_metapy(tweet):
    doc = metapy.index.Document()
    doc.content(tweet)
    # Create tokenizer (tags sentance boundaries)
    tok = metapy.analyzers.ICUTokenizer(suppress_tags=True)
    # Lowercase all tokens
    tok = metapy.analyzers.LowercaseFilter(tok)
    # Remove tokens that are less than 2 or more than 30 characters (punctuation, URLs)
    tok = metapy.analyzers.LengthFilter(tok, min=2, max=30)
    # Stemming
    tok = metapy.analyzers.Porter2Filter(tok)
    # POS tagging
    # tagger = metapy.sequence.PerceptronTagger("perceptron-tagger/")
    # tok = metapy.analyzers.PennTreebankNormalizer(tok)
    # seqs = []
    # for seq in extract_sequences(tok):
    #     tagger.tag(seq)
    #     print(seq)
    #     seqs.append(seq)
    tok.set_content(doc.content())
    tokens = [token for token in tok]
    # print(seqs)

    return tokens

def get_ngram_list(tokens, n):
    tokens = [t for t in tokens if not t.startswith('#')]
    tokens = [t for t in tokens if not t.startswith('@')]
    ngram_list = [gram for gram in ngrams(tokens, n)]
    return ngram_list

def get_ngram_features(ngrams, ngram_map):
    feature_list = [0] * np.zeros(len(ngram_map))
    for gram in ngrams:
        if gram in ngram_map:
            feature_list[ngram_map[gram]] += 1.0
    return feature_list

def get_ngrams(tokens, n, stopwords):
    if len(n) < 1:
        return {}
    filtered = []
    # stopwords = data_proc.get_stopwords_list()
    for t in tokens:
        if t not in stopwords and t.isalnum():
            filtered.append(t)
    tokens = filtered
    ngram_features = {}
    for i in n:
        ngram_list = [gram for gram in ngrams(tokens, i)]
        ngram_features[i] = ngram_list
    return np.array(ngram_features)

def create_ngram_columns(df):
    df["unigrams"] = df["tokens"].apply(get_ngram_list, args=[1])
    df["bigrams"] = df["tokens"].apply(get_ngram_list, args=[2])
    df["trigrams"] = df["tokens"].apply(get_ngram_list, args=[3])

    return df

def final_ngram_cols(df, ngram_map):
    df["unigram_features"] = df["unigrams"].apply(get_ngram_features, args=[ngram_map])
    df["bigram_features"] = df["bigrams"].apply(get_ngram_features, args=[ngram_map])
    df["trigram_features"] = df["trigrams"].apply(get_ngram_features, args=[ngram_map])
    df["ngram_features"] = df["unigram_features"] + df["bigram_features"] + df["trigram_features"]

    ngram_cols = ["n-gram-" + str(i) for i in range(len(ngram_map))]
    df[ngram_cols] = pd.DataFrame(df.ngram_features.tolist(), index=df.index)

    return df

def get_simple_features(df):
    # remove @USER from responses
    # Ger number of users tagged in reponse
    df["users_tagged"] = df.response.str.count("@USER")
    df["response"] = df.response.str.replace('@USER', '').str.strip()
    df["tokens"] = df["response"].str.split()

    df["num_hashtags"] = df.response.str.count("#")
    df["num_capital"] = df.response.str.findall(r'[A-Z]').str.len()
    # Currently includes punctuation
    df["tweet_length_words"] = df.tokens.str.len()
    df["tweet_length_char"] = df.tokens.apply(lambda tweet: sum([len(w) for w in tweet]))
    df["average_token_length"] = df["tweet_length_words"].astype(float) / df["tweet_length_char"]

    df["contains_laughter"] = df.tokens.apply(
        lambda tweet: len([1 for w in tweet if (w.lower().startswith("haha")) or (re.match('l(o)+l$', w.lower()))])
    )
    df["contains_ellipses"] = df.tokens.apply(lambda tweet: len([w for w in tweet if (w == '...') | (w == '..')]))
    df["strong_negations"] = df.tokens.apply(lambda tweet: len([w for w in tweet if w in helper.strong_negations]))
    df["strong_affirmatives"] = df.tokens.apply(lambda tweet: len([w for w in tweet if w in helper.strong_affirmatives]))
    df["interjections"] = df.tokens.apply(lambda tweet: len([w for w in tweet if w in helper.interjections]))
    df["intensifiers"] = df.tokens.apply(lambda tweet: len([w for w in tweet if w in helper.intensifiers]))
    df["punctuation"] = df.tokens.apply(lambda tweet: len([w for w in tweet if w in helper.punctuation]))
    df["emojis"] = df.tokens.apply(lambda tweet: len([w for w in tweet if w in emoji.UNICODE_EMOJI]))

    # Clean up responses for tokenizing
    df["tokens"] = df["response"].apply(run_metapy)
    return df

# Enter mutable info

data_dir = os.path.dirname(os.getcwd()) + '/data'
filename = 'train.jsonl'
file = os.path.join(data_dir, filename)  # .json file
df = utils.parse_json(file)

test_filename = 'test.jsonl'
test_file = os.path.join(data_dir, test_filename)  # .json file
test_df = utils.parse_json(test_file)

df = get_simple_features(df)

test_df = get_simple_features(test_df)

# need to remove punctionation, emojis before doing this
unigrams = Counter()
bigrams = Counter()
trigrams = Counter()
df = create_ngram_columns(df)

test_df = create_ngram_columns(test_df)

df["unigrams"].apply(unigrams.update)
df["bigrams"].apply(bigrams.update)
df["trigrams"].apply(trigrams.update)


# ngrams = get_ngrams(tokens, ngram_len, stopword_list)
# print(ngrams)
# unigrams = Counter(ngrams[1])
# bigrams = Counter(ngrams[2])
# trigrams = Counter(ngrams[3])
#
unigram_tokens = [k for k, c in unigrams.items() if c > 2]
bigram_tokens = [k for k, c in bigrams.items() if c > 2]
trigram_tokens = [k for k, c in trigrams.items() if c > 2]
# print(unigram_tokens)
# print(bigram_tokens)
# print(trigram_tokens)
#
ngram_map = dict()
all_ngrams = unigram_tokens
all_ngrams.extend(bigram_tokens)
all_ngrams.extend(trigram_tokens)
for i in range(0, len(all_ngrams)):
    ngram_map[all_ngrams[i]] = i
# print(ngram_map)

df = final_ngram_cols(df, ngram_map)
test_df = final_ngram_cols(test_df, ngram_map)

# df["POS"] = df.response.apply(run_metapy)
df.drop(columns=["context", "tokens", "response", "unigrams", "bigrams", "trigrams", "unigram_features", "bigram_features", "trigram_features", "ngram_features"], inplace=True)
test_df.drop(columns=["context", "tokens", "response", "unigrams", "bigrams", "trigrams", "unigram_features", "bigram_features", "trigram_features", "ngram_features"], inplace=True)

print(df.shape)
print(df.columns)
print(test_df.shape)
print(test_df.columns)


df.to_csv("train_feature_engineering.csv", index=False)
test_df.to_csv("test_feature_engineering.csv", index=False)
