import os
import re
from jx_dev.utils_text_clf import utils_text_clf as utils

import warnings
warnings.filterwarnings("ignore")

# Enter mutable info

data_dir = os.path.dirname(os.getcwd()) + '/data'
filename = 'train.jsonl'
file = os.path.join(data_dir, filename)  # .json file

# parse the .json file

df = utils.parse_json(file)
print(df.head())

# remove @USER from responses
# Ger number of users tagged in reponse
df["users_tagged"] = df.response.str.count("@USER")
df["response"] = df.response.str.replace('@USER', '').str.strip()
df["tokens"] = df["response"].str.split()

df["num_hashtags"] = df.response.str.count("#")
df["num_capital"] = df.response.str.findall(r'[A-Z]').str.len()
# Currently includes punctuation
df["tweet_length_words"] = df.tokens.str.len()

df["contains_laughter"] = df.tokens.apply(
    lambda tweet: len([1 for w in tweet if (w.lower().startswith("haha")) or (re.match('l(o)+l$', w.lower()))])
)

df.to_csv("train_feature_engineering.csv", index=False)
