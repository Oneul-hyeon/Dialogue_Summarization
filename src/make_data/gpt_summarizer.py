import os
import time
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

def get_summary(idx, dialogue) :
    flag = True
    while flag :
        try :
            summary = chain.invoke({"dialogue" : dialogue})
            flag = False
        except Exception as e:
            print(f'{idx} error : {str(e)}')
            time.sleep(1)

    return dict(summary)

with open(f'{os.getcwd()}/key/Tier3_API_Key.txt', 'r') as f :
    TIER3_API_KEY = f.read()
os.environ["OPENAI_API_KEY"] = TIER3_API_KEY

llm = ChatOpenAI(model = "gpt-4o", temperature = 0.2)

class Summary(BaseModel) :
    summary : str = Field(description = "해당 민원 내용의 요약")

parser = PydanticOutputParser(pydantic_object = Summary)

prompt = PromptTemplate.from_template(""" 
너는 민원의 내용을 최대 2문장으로 요약하는 민원 요약기야.
입력되는 민원에 대한 요약을 해줘.
이 요약은 해당 민원을 진행했던 다른 상담사가 볼 거야.
그러니까 이 민원 요약은 다른 상담사가 처음 봐도 이해할 수 있는 민원 요약이어야 해.
요약의 어투는 상냥하게 해 주고 대화의 끝은 상담사가 어떻게 안내했는지까지 알려줘.
답변은 아래의 예시처럼 제공해줘야 해.
아래의 예시처럼 하는데 제공하는 건 dialogue는 빼고 summary만 제공해 줘.
                                      
["dialogue" : '''혹시 여기가 상담 센터 맞나요? \
여기가 맞습니다만 무슨일로 연락 하신건가요? \
대중교통 관련하여 질문이 있어서 연락 드렸습니다. \
궁금한게 뭔가요? 성남 시내버스 첫차가 몇 시 인가요? \
질문의 이유가 무엇인가요? \
아 그냥 알고 싶어서요. \
알겠습니다. 잠시만 정보를 찾아봐도 될까요? \
네 천천히 하세요. \
노선마다 다르지만 성남 시내버스는 가장 빠른게 05시25분 입니다. \
아 몰랐던 사실이네요. \
충분한 대답이 됐나요? \
네 이해가 잘 됩니다. \
다행입니다. 또 다른 질문 있나요? \
여줘보고 싶은데 하나 더 있습니다. \
아 그렇군요. 편하게 말씀하세요 \
그럼 성남 시내버스 막차는 몇 시 인가요? \
노선마다 다르지만 성남 시내버스는 가장 늦는게 23시35분 입니다. \
그렇군요. 제 질문은 여기까지 입니다. 감사합니다. \
별 말씀을요. 궁금한거 있으면 언제든 문의 해 주세요.''',
"summary" : 성남 시내버스 첫차와 막차가 몇 시인지 문의하여 노선마다 다르지만 가장 빠른게 05시25분이고 가장 늦는게 23시35분이라고 안내했습니다.'],
["dialogue" : '''청구서 납부 문의 드립니다. \
수도요금 납부하시는 건가요? \
은행에서 납부 가능한가요? \
창구에서 납부하시면 됩니다. \
기계로도 납부 할수 있나요? \
공과금 전용수납기를 통해서 납부 가능합니다. \
모든 은행에서 납부 할수 있는건가요? \
네. 은행에서 납부 가능합니다. \
납부시간은 어떻게 되나요? \
09:00-16:00까지 입니다. \
자동이체 가능한가요? \
네. 자동이체 가능합니다. \
더 궁금하신 사항 있으신가요? \
급수설비 폐지 신청 문의 드립니다. \
급수설비 폐지하려는 건가요? \
네. 재건축할꺼라서요. \
인터넷으로 신청할수 있나요? \
네. 인터넷으로 신청하시면 됩니다. \
처리하는데 얼마나 걸릴까요? \
5일정도 소요됩니다.''',
"summary" : 수도요금 납부에 대해 문의하여 창구에서 납부하거나 공과금 전용수납기를 통해 납부 가능하다고 안내했고, 급수설비 폐지 신청에 대해 문의하여 인터넷으로 신청하면 5일정도 소요된다고 안내했습니다.']

이 대화를 위 예시와 같이 'summary'라는 키를 가진 JSON 객체로 반환해 줘

민원 상담 내용 :
{dialogue}

JSON 출력 :
{format}
""")

prompt = prompt.partial(format = parser.get_format_instructions())
chain = prompt | llm | parser