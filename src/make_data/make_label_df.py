import os
import json
import pandas as pd

os.chdir("..")

RAW_DIR = f"{os.getcwd()}/data/raw"
JSON_DIR = f"{os.getcwd()}/data/json"
JSON_TRAIN_DIR = JSON_DIR + "/train"
JSON_TEST_DIR = JSON_DIR + "/test"
SAVE_DIR = f"{os.getcwd()}/data/labeled_df"

train_df = pd.read_csv(f"{RAW_DIR}/train_df.csv")
test_df = pd.read_csv(f"{RAW_DIR}/test_df.csv")

train_summary = []
for file in sorted(os.listdir(JSON_TRAIN_DIR), key = lambda x : int(x.split(".")[0])) :
    with open(f"{JSON_TRAIN_DIR}/{file}", "r") as f :
        json_data = json.load(f)
    train_summary.append(json_data["summary"])
train_df["summary"] = train_summary

test_summary = []
for file in sorted(os.listdir(JSON_TEST_DIR), key = lambda x : int(x.split(".")[0])) :
    with open(f"{JSON_TEST_DIR}/{file}", "r") as f :
        json_data = json.load(f)
    test_summary.append(json_data["summary"])
test_df["summary"] = test_summary

train_df.to_csv(f"{SAVE_DIR}/labeled_train_df.csv", index = False)
test_df.to_csv(f"{SAVE_DIR}/labeled_test_df.csv", index = False)