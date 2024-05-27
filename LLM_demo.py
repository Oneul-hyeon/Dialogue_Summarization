import gradio as gr
import pandas as pd
from ast import literal_eval
from src.evaluate import get_summary, get_sub_summary
import pickle
import random
import requests, json

def find_dialogue(dialogue_number) :
    dialogue = list(test_df[test_df["dialogue_number"] == dialogue_number]["dialogue"])[0]
    return "\n".join(dialogue)

def find_label(dialogue_number) :
    label = list(KoBART_test_df[KoBART_test_df["dialogue_number"] == dialogue_number]["summary"])[0]
    return label

def preprocess(dialogue_) :
    utter = ""
    split_text = list(dialogue_.split("\n"))

    for utterance in split_text :
        idx = utterance.index(":")
        utter_ = utterance[idx+1:].lstrip()
        utter += ' ' + utter_
    return utter.lstrip()

def get_llm_summary(str_dialogue):
    URL = "http://192.168.123.32:9706/respond/summary"
    data = {"message": str_dialogue}
    response = requests.post(URL, data=json.dumps(data))
    result_j = response.text
    result_object = json.loads(result_j)
    print(str_dialogue)
    return result_object['context']
    
def predict_summary(dialogue_number, dialogue) :
    str_dialogue = preprocess(dialogue)
    label = find_label(dialogue_number)
    summary = get_summary(str_dialogue)
    sub_summary = get_sub_summary(str_dialogue)
    llm_summary = get_llm_summary(str_dialogue)
    return label, summary, sub_summary, llm_summary

test_df = pd.read_csv("data/raw/test_df.csv")
test_df["dialogue"] = test_df.dialogue.apply(lambda x : literal_eval(x))
KoBART_test_df = pd.read_csv("data/KoBART_df/kobart_test_df.csv")

# 마크다운 내용
markdown_content = """
A~ : 금융/보험\n
B~ : 다산콜센터
"""

with open("data/example_data/example_A.pkl", "rb") as f :
    example_A = pickle.load(f)
with open("data/example_data/example_B.pkl", "rb") as f :
    example_B = pickle.load(f)  

# 레이아웃 정의 및 실행
with gr.Blocks() as demo:
    # 인터페이스 타이틀
    gr.Markdown("# KoBART 기반 다산 콜센터, 금융/보험 관련 생성 요약")
    # 타이틀 바로 아래에 마크다운을 배치
    gr.Markdown(markdown_content)

    with gr.Row() :
        with gr.Column() :
            # 입력 및 출력 컴포넌트 정의
            input_text = gr.Textbox(label="dialogue number")
            output_dialogue = gr.Textbox(label="dialogue", interactive = True)
        with gr.Column() :
            # 예제들 추가
            gr.Examples(examples=example_A, inputs=input_text, label = "금융/보험 도메인 예제")
            # 예제들 추가
            gr.Examples(examples=example_B, inputs=input_text, label = "다산콜센터 예제")
    btn = gr.Button("예제 출력")
    btn.click(fn=find_dialogue, inputs=input_text, outputs=[output_dialogue])

    gpt4o_summary = gr.Textbox(label="GPT-4o기반 정답 요약")
    output_summary = gr.Textbox(label="요약 결과(freeze)")
    output_sub_summary = gr.Textbox(label="요약 결과(today)")
    output_llm_summary = gr.Textbox(label="Local LLM요약 결과")
    # 버튼을 눌렀을 때 실행할 함수와 연결
    btn = gr.Button("요약")
    btn.click(fn=predict_summary, inputs=[input_text, output_dialogue], outputs=[gpt4o_summary, output_summary, output_sub_summary, output_llm_summary])

demo.launch(server_name="0.0.0.0", server_port=11030)