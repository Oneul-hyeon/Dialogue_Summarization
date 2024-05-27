import pandas as pd
from transformers import GPT2TokenizerFast
from ast import literal_eval

def cal_tok(dialogue) :
    return sum([len(tokenizer.tokenize(utterance)) for utterance in dialogue])

train_df = pd.read_csv("train_df.csv")
test_df = pd.read_csv("test_df.csv")

tokenizer = GPT2TokenizerFast.from_pretrained("Xenova/gpt-4o")

train_df["dialogue"] = train_df.dialogue.apply(lambda x : literal_eval(x))
test_df["dialogue"] = test_df.dialogue.apply(lambda x : literal_eval(x))

train_df["token_len"] = train_df.dialogue.apply(lambda x : cal_tok(x))
test_df["token_len"] = test_df.dialogue.apply(lambda x : cal_tok(x))

print(f'train token len : {sum(train_df["token_len"])} | test token len : {sum(test_df["token_len"])}')
print(f'train Min token len : {min(train_df["token_len"])} | train Max token len : {max(train_df["token_len"])} | test Min token len : {min(test_df["token_len"])} | test Max token len : {max(test_df["token_len"])}')
print(train_df["token_len"].describe(percentiles = [0.25, 0.50, 0.75, 0.95]))
print(test_df["token_len"].describe(percentiles = [0.25, 0.50, 0.75, 0.95]))