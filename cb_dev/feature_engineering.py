import emoji
import metapy
from nltk import ngrams, pos_tag
from nltk.tokenize import TweetTokenizer, RegexpTokenizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import sentiwordnet as swn
import itertools
import os
import re
import numpy as np
import pandas as pd
from jx_dev.utils_text_clf import utils_text_clf as utils
import cb_dev.vocab_helpers as helper
from collections import Counter


import warnings
warnings.filterwarnings("ignore")
# Get sentiment features -- a total of 16 features derived
# Emoji features: a count of the positive, negative and neutral emojis
# along with the ratio of positive to negative emojis and negative to neutral
# Using the MPQA subjectivity lexicon, we have to check words for their part of speech
# and obtain features: a count of positive, negative and neutral words, as well as
# a count of the strong and weak subjectives, along with their ratios and a total sentiment words.
# Also using VADER sentiment analyser to obtain a score of sentiments held in a tweet (4 features)
def get_sentiment_features(df, emoji_sent_dict):
    tweet = df.reponse
    tweet_tokens = df.tokens
    tweet_pos = df.tweet_pos

    # sent_features = dict.fromkeys(["positive emoji", "negative emoji", "neutral emoji",
    #                                "subjlexicon weaksubj", "subjlexicon strongsubj",
    #                                "subjlexicon positive", "subjlexicon negative",
    #                                "subjlexicon neutral", "total sentiment words",
    #                                "swn pos", "swn neg", "swn obj"], 0.0)
    for t in tweet_tokens:
        if t in emoji_sent_dict.keys():
            df['negative emoji'] += float(emoji_sent_dict[t][0])
            df['neutral emoji'] += float(emoji_sent_dict[t][1])
            df['positive emoji'] += float(emoji_sent_dict[t][2])

    lemmatizer = WordNetLemmatizer()
    pos_translation = {'N': 'n', 'V': 'v', 'D': 'a', 'R': 'r'}
    for index in range(len(tweet_tokens)):
        lemmatized = lemmatizer.lemmatize(tweet_tokens[index], 'v')
        if tweet_pos[index] in pos_translation:
            synsets = list(swn.senti_synsets(lemmatized, pos_translation[tweet_pos[index]]))
            pos_score = 0
            neg_score = 0
            obj_score = 0
            if len(synsets) > 0:
                for syn in synsets:
                    pos_score += syn.pos_score()
                    neg_score += syn.neg_score()
                    obj_score += syn.obj_score()
                df["swn pos"] = pos_score / float(len(synsets))
                df["swn neg"] = neg_score / float(len(synsets))
                df["swn obj"] = obj_score / float(len(synsets))

    # Vader Sentiment Analyser
    # Obtain the negative, positive, neutral and compound scores of a tweet
    sia = SentimentIntensityAnalyzer()
    polarity_scores = sia.polarity_scores(tweet)
    for name, score in polarity_scores.items():
        df["Vader score " + name] = score
    return df



# Split based on Camel Case
def camel_case_split(term):
    term = re.sub(r'([0-9]+)', r' \1', term)
    term = re.sub(r'(1st|2nd|3rd|4th|5th|6th|7th|8th|9th|0th)', r'\1 ', term)
    splits = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', term)
    return [s.group(0) for s in splits]

def split_hashtags(hashtag, word_list, verbose=False):
    if verbose:
        print("Hashtag is %s" % hashtag)
    # Get rid of the hashtag
    if hashtag.startswith('#'):
        term = hashtag[1:]
    else:
        term = hashtag

    # If the hastag is already an existing word (a single word), return it
    if word_list is not None and term.lower() in word_list:
        return ['#' + term]
    # First, attempt splitting by CamelCase
    if term[1:] != term[1:].lower() and term[1:] != term[1:].upper():
        splits = camel_case_split(term)
    elif '#' in term:
        splits = term.split("#")
    elif len(term) > 27:
        if verbose:
            print("Hashtag %s is too big so let as it is." % term)
        splits = [term]
    else:
        # Second, build possible splits and choose the best split by assigning
        # a "score" to each possible split, based on the frequency with which a word is occurring
        penalty = -69971
        max_coverage = penalty
        max_splits = 6
        n_splits = 0
        term = re.sub(r'([0-9]+)', r' \1', term)
        term = re.sub(r'(1st|2nd|3rd|4th|5th|6th|7th|8th|9th|0th)', r'\1 ', term)
        term = re.sub(r'([A-Z][^A-Z ]+)', r' \1', term.strip())
        term = re.sub(r'([A-Z]{2,})+', r' \1', term)
        splits = term.strip().split(' ')
        if len(splits) < 3:
            # Splitting lower case and uppercase hashtags in up to 5 words
            chars = [c for c in term.lower()]
            found_all_words = False

            while n_splits < max_splits and not found_all_words:
                for index in itertools.combinations(range(0, len(chars)), n_splits):
                    output = np.split(chars, index)
                    line = [''.join(o) for o in output]
                    score = 0.0
                    for word in line:
                        stripped = word.strip()
                        if stripped in word_list:
                            score += int(word_list.get(stripped))
                        else:
                            if stripped.isnumeric():  # not stripped.isalpha():
                                score += 0.0
                            else:
                                score += penalty
                    score = score / float(len(line))
                    if score > max_coverage:
                        splits = line
                        max_coverage = score
                        line_is_valid_word = [word.strip() in word_list if not word.isnumeric()
                                              else True for word in line]
                        if all(line_is_valid_word):
                            found_all_words = True
                n_splits = n_splits + 1
    splits = ['#' + str(s) for s in splits]
    if verbose:
        print("Split to: ", splits)
    return splits

# Initial tweet cleaning - useful to filter data before tokenization
def clean_tweet(tweet, word_list, split_hashtag_method=split_hashtags, replace_user_mentions=True,
                remove_hashtags=False, remove_emojis=False, all_to_lower_case=False):
    # Add white space before every punctuation sign so that we can split around it and keep it
    tweet = re.sub('([!?*&%"~`^+{}])', r' \1 ', tweet)
    tweet = re.sub('\s{2,}', ' ', tweet)
    tokens = tweet.split()
    valid_tokens = []
    for word in tokens:
        # Never include #sarca* hashtags
        if word.lower().startswith('#sarca'):
            continue
        # Never include URLs
        if 'http' in word:
            continue
        # Replace specific user mentions with a general user name
        if replace_user_mentions and word.startswith('@'):
            word = '@user'
        # Split or remove hashtags
        if word.startswith('#'):
            if remove_hashtags:
                continue
            splits = split_hashtag_method(word[1:], word_list)
            if all_to_lower_case:
                valid_tokens.extend([split.lower() for split in splits])
            else:
                valid_tokens.extend(splits)
            continue
        if remove_emojis and word in emoji.UNICODE_EMOJI:
            continue
        if all_to_lower_case:
            word = word.lower()
        valid_tokens.append(word)
    return ' '.join(valid_tokens)


def make_normalizer(mean, std):
    def normalize(x):
        return (x - mean)/std
    return normalize


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
    df["ngram_feature"] = df[ngram_cols].sum(axis=1)
    df.drop(columns=ngram_cols, inplace=True)

    return df

def get_simple_features(df, emoji_sent_dict):
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
    df['negative emoji'] = df.tokens.apply(lambda tweet: sum([float(emoji_sent_dict[t][0]) for t in tweet if t in emoji_sent_dict.keys()]))
    df['neutral emoji'] = df.tokens.apply(lambda tweet: sum([float(emoji_sent_dict[t][1]) for t in tweet if t in emoji_sent_dict.keys()]))
    df['positive emoji'] = df.tokens.apply(lambda tweet: sum([float(emoji_sent_dict[t][2]) for t in tweet if t in emoji_sent_dict.keys()]))

    # Clean up responses for tokenizing
    df["tokens"] = df["response"].apply(run_metapy)
    return df

def context_features(df, emoji_sent_dict):
    # print(df)
    users_tagged = 0
    tokens = []

    num_hashtags = 0
    num_capital = 0
    # Currently includes punctuation
    tweet_length_words = 0
    tweet_length_char = 0
    average_token_length = 0

    contains_laughter = 0
    contains_ellipses = 0
    strong_negations = 0
    strong_affirmatives = 0
    interjections = 0
    intensifiers = 0
    punctuation = 0
    emojis = 0
    negative_emoji = 0
    neutral_emoji = 0
    positive_emoji = 0

    for tweet in df.loc['context']:
        users_tagged += tweet.count("@USER")
        tokens.append(tweet.split())

        num_hashtags += tweet.count("#")
        num_capital += tweet.count(r'[A-Z]')
        # Currently includes punctuation
        tweet_length_words += len(tweet.split())
        tweet_length_char += len(tweet)
        average_token_length += len(tweet.split())/len(tweet)
        for w in tweet:
            if (w.lower().startswith("haha")) or (re.match('l(o)+l$', w.lower())):
                contains_laughter += 1

        contains_ellipses += len([w for w in tweet if (w == '...') | (w == '..')])
        strong_negations += len([w for w in tweet if w in helper.strong_negations])
        strong_affirmatives += len([w for w in tweet if w in helper.strong_affirmatives])
        interjections += len([w for w in tweet if w in helper.interjections])
        intensifiers += len([w for w in tweet if w in helper.intensifiers])
        punctuation += len([w for w in tweet if w in helper.punctuation])
        emojis += len([w for w in tweet if w in emoji.UNICODE_EMOJI])
        negative_emoji += sum([float(emoji_sent_dict[t][0]) for t in tweet if t in emoji_sent_dict.keys()])
        neutral_emoji += sum([float(emoji_sent_dict[t][1]) for t in tweet if t in emoji_sent_dict.keys()])
        positive_emoji += sum([float(emoji_sent_dict[t][2]) for t in tweet if t in emoji_sent_dict.keys()])
    df["context_users_tagged"] = users_tagged
    df["context_tokens"] = tokens
    df["context_num_hashtags"] = num_hashtags
    df["context_num_capital"] = num_capital
    df["context_tweet_length_words"] = tweet_length_words
    df["context_tweet_length_char"] = tweet_length_char
    df["context_average_token_length"] = average_token_length
    df["context_contains_laughter"] = contains_laughter
    df["context_contains_ellipses"] = contains_ellipses
    df["context_strong_negations"] = strong_negations
    df["context_strong_affirmatives"] = strong_affirmatives
    df["context_interjections"] = interjections
    df["context_intensifiers"] = intensifiers
    df["context_punctuation"] = punctuation
    df["context_emojis"] = emojis
    df['context_negative emoji'] = negative_emoji
    df['context_neutral emoji'] = neutral_emoji
    df['context_positive emoji'] = positive_emoji

    return df

if __name__ == "__main__":
    stopword_list = []
    word_list = {}
    emoji_sent_dict = {}
    with open("stopwords.txt", 'r') as file:
        text = file.read()
        file.close()
        stopword_list = text.split("\n")

    with open("word_list.txt", 'r') as file:
        lines = file.read()
        file.close()
        for line in lines.split("\n"):
            key, value = line.split("\t")
            word_list[key] = value
    with open("emoji_sent_dict.txt", 'r') as file:
        lines = file.read()
        file.close()
        for line in lines.split("\n"):
            key = line.split("\t")[0]
            value = line.split("\t")[1:]
            emoji_sent_dict[key] = value

    data_dir = os.path.dirname(os.getcwd()) + '/data'
    filename = 'train.jsonl'
    file = os.path.join(data_dir, filename)  # .json file
    train_df = utils.parse_json(file)

    test_filename = 'test.jsonl'
    test_file = os.path.join(data_dir, test_filename)  # .json file
    test_df = utils.parse_json(test_file)

    train_df["response"] = train_df.response.apply(clean_tweet, args=[word_list])
    test_df["response"] = test_df.response.apply(clean_tweet, args=[word_list])

    train_df = get_simple_features(train_df, emoji_sent_dict)

    test_df = get_simple_features(test_df, emoji_sent_dict)

    # need to remove punctionation, emojis before doing this
    unigrams = Counter()
    bigrams = Counter()
    trigrams = Counter()
    train_df = create_ngram_columns(train_df)

    test_df = create_ngram_columns(test_df)

    train_df["unigrams"].apply(unigrams.update)
    train_df["bigrams"].apply(bigrams.update)
    train_df["trigrams"].apply(trigrams.update)

    unigram_tokens = [k for k, c in unigrams.items() if c > 2]
    bigram_tokens = [k for k, c in bigrams.items() if c > 2]
    trigram_tokens = [k for k, c in trigrams.items() if c > 2]

    ngram_map = dict()
    all_ngrams = unigram_tokens
    all_ngrams.extend(bigram_tokens)
    all_ngrams.extend(trigram_tokens)
    for i in range(0, len(all_ngrams)):
        ngram_map[all_ngrams[i]] = i

    train_df = final_ngram_cols(train_df, ngram_map)
    test_df = final_ngram_cols(test_df, ngram_map)

    train_df = train_df.apply(context_features, axis=1, args=[emoji_sent_dict])
    test_df = test_df.apply(context_features, axis=1, args=[emoji_sent_dict])

    cols_to_drop = [
        "context",
        "tokens",
        "response",
        "unigrams",
        "bigrams",
        "trigrams",
        "unigram_features",
        "bigram_features",
        "trigram_features",
        "ngram_features",
        "context_tokens"
    ]
    train_df.drop(columns=cols_to_drop, inplace=True)
    test_df.drop(columns=cols_to_drop, inplace=True)

    # Normalize:
    train_label = train_df.pop('label')
    test_label = test_df.pop('id')
    train_stats = train_df.describe()
    print(train_stats.columns)
    print(train_df.columns)
    # train_stats.pop('label')
    train_stats = train_stats.transpose()
    norm = make_normalizer(train_stats["mean"], train_stats["std"])

    print(train_df.dtypes)
    print(test_df.dtypes)

    normed_train_data = norm(train_df)
    normed_test_data = norm(test_df)
    # normed_train_data = train_df
    # normed_test_data = test_df

    normed_train_data.insert(loc=0, column='label', value=train_label)
    normed_train_data.fillna(0, inplace=True)
    normed_test_data.insert(loc=0, column='id', value=test_label)
    normed_test_data.fillna(0, inplace=True)

    print(normed_train_data.shape)
    print(normed_train_data.columns)
    print(normed_test_data.shape)
    print(normed_test_data.columns)


    normed_train_data.to_csv("data/train_feature_engineering.csv", index=False)
    normed_test_data.to_csv("data/test_feature_engineering.csv", index=False)
