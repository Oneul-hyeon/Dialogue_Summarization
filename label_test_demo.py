import gradio as gr
import pandas as pd
import numpy as np
from ast import literal_eval

def view_data() :
    idx = np.random.choice(range(l), 1)[0]
    row = train_df[train_df.index == idx]
    dialogue = '\n'.join(*row["dialogue"])
    summary = row["summary"][idx]
    return dialogue, summary

train_df = pd.read_csv("data/labeled_df/labeled_train_df.csv")
train_df["dialogue"] = train_df.dialogue.apply(lambda x : literal_eval(x))

l = len(train_df.index)
demo = gr.Interface(fn=view_data,
                    inputs = None,
                    outputs = [gr.Textbox(label = "dialogue"), gr.Textbox(label = "summary")],
                    title = "GPT-4o 기반 라벨 생성")

demo.launch(server_name = "0.0.0.0", server_port = 11028)