import gradio as gr
import json
from make_data.gpt_summarizer import get_summary

def split_utterance(utterance) :
    idx = utterance.index(":")
    new_utterance = utterance[idx+1:].strip()
    return new_utterance

def make_label_summary(dialogue) :
    new_dialogue = ' '.join([split_utterance(utterance) for utterance in dialogue.split("\n")])
    print(new_dialogue)
    print()
    summary = get_summary(new_dialogue)
    return summary["summary"]

demo = gr.Interface(fn=make_label_summary,
                    inputs = gr.Textbox(),
                    outputs = [gr.Textbox(label = "생성된 Label")],
                    title = "GPT-4o 기반 라벨 생성")

demo.launch(server_name = "0.0.0.0", server_port = 11027)