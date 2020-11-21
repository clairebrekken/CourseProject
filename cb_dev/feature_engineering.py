import emoji
import metapy
import os
import re
from jx_dev.utils_text_clf import utils_text_clf as utils
import cb_dev.vocab_helpers as helper

import warnings
warnings.filterwarnings("ignore")

def extract_sequences(tok):
    sequences = []
    for token in tok:
        if token == '<s>':
            sequences.append(metapy.sequence.Sequence())
        elif token != '</s>':
            sequences[-1].add_symbol(token)
    return sequences

def run_metapy(tweet):
    print(tweet)
    doc = metapy.index.Document()
    doc.content(tweet)
    # Create tokenizer (tags sentance boundaries)
    tok = metapy.analyzers.ICUTokenizer()
    # Lowercase all tokens
    tok = metapy.analyzers.LowercaseFilter(tok)
    # Remove tokens that are less than 2 or more than 30 characters (punctuation, URLs)
    tok = metapy.analyzers.LengthFilter(tok, min=2, max=30)
    # Stemming
    tok = metapy.analyzers.Porter2Filter(tok)
    # POS tagging
    tagger = metapy.sequence.PerceptronTagger("perceptron-tagger/")
    # tok = metapy.analyzers.PennTreebankNormalizer(tok)
    seqs = []
    for seq in extract_sequences(tok):
        tagger.tag(seq)
        print(seq)
        seqs.append(seq)
    tok.set_content(doc.content())
    tokens = [token for token in tok]
    print(seqs)
    print(tokens)

    return tokens


# Enter mutable info

data_dir = os.path.dirname(os.getcwd()) + '/data'
filename = 'test.jsonl'
file = os.path.join(data_dir, filename)  # .json file

# parse the .json file

df = utils.parse_json(file)
df = df.head()

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

# df["POS"] = df.response.apply(run_metapy)
df.drop(columns=["context", "tokens", "response"], inplace=True)

df.to_csv("test_feature_engineering.csv", index=False)
