import os
import pandas as pd

from make_data.gpt_summarizer import get_summary
from ast import literal_eval
from tqdm import tqdm
import json

def split_utterance(utterance) :
    idx = utterance.index(":")
    new_utterance = utterance[idx+1:].strip()
    return new_utterance

def make_label_summary(mode, df) :
    for idx, row in tqdm(df.iterrows()) :
        if mode == "train" and idx <= 1428 : continue

        dialogue = row["dialogue"]
        split_dialogue = ' '.join([split_utterance(utterance) for utterance in dialogue])
        summary = get_summary(idx, split_dialogue)
        with open(f'data/json/{mode}/{idx}.json', "w", encoding = "utf-8") as f:
            json.dump(summary, f, ensure_ascii= False) 

os.chdir("..")

train_df = pd.read_csv(f'{os.getcwd()}/data/raw/train_df.csv')
test_df = pd.read_csv(f'{os.getcwd()}/data/raw/test_df.csv')

train_df["dialogue"] = train_df.dialogue.apply(lambda x : literal_eval(x))
test_df["dialogue"] = test_df.dialogue.apply(lambda x : literal_eval(x))

print("Start train df Labeling...")
make_label_summary("train", train_df)
print("End train df Labeling...")

print("Start test df Labeling...")
make_label_summary("test", test_df)
print("End test df Labeling...")