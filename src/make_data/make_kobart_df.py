import os
import pandas as pd
from sklearn.model_selection import train_test_split
from ast import literal_eval
os.chdir("..")
os.chdir("..")

def preprocess(dialogue_) :
    dialogue = []
    for utterance in dialogue_ :
        idx = utterance.index(":")
        utter = utterance[idx+1:]
        dialogue.append(utter)
    return dialogue

labeled_train_df = pd.read_csv("data/labeled_df/labeled_train_df.csv")
labeled_test_df = pd.read_csv("data/labeled_df/labeled_test_df.csv")

labeled_train_df["dialogue"] = labeled_train_df.dialogue.apply(lambda x : literal_eval(x))
labeled_test_df["dialogue"] = labeled_test_df.dialogue.apply(lambda x : literal_eval(x))

# make kobart_train_df / valid_df
kobart_train_df = labeled_train_df[["dialogue", "summary"]]
kobart_train_df["dialogue"] = kobart_train_df.dialogue.apply(lambda x : ' '.join(preprocess(x)))
# kobart_train_df, kobart_valid_df = train_test_split(kobart_train_df, test_size=0.1, random_state=42, shuffle = True)
kobart_train_df.to_csv("data/KoBART_df/kobart_train_df.csv", index = False)
# kobart_valid_df.to_csv("data/KoBART_df/kobart_valid_df.csv", index = False)

# make kobart_test_df
kobart_test_df = labeled_test_df[["dialogue", "summary"]]
kobart_test_df["dialogue"] = kobart_test_df.dialogue.apply(lambda x : ' '.join(preprocess(x)))
kobart_test_df.to_csv("data/KoBART_df/kobart_test_df.csv", index = False)