import torch
import pandas as pd
from transformers import PreTrainedTokenizerFast
from transformers.models.bart import BartForConditionalGeneration

def get_summary(dialogue) :
    input_ids = tokenizer.encode(dialogue)
    input_ids = torch.tensor(input_ids)
    input_ids = input_ids.unsqueeze(0).to(device)
    output = model.generate(input_ids, eos_token_id = 1, max_length = 512, num_beams = 5)
    output = tokenizer.decode(output[0], skip_special_tokens = True)
    return output

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BartForConditionalGeneration.from_pretrained('./kobart_summary')
model.to(device)
tokenizer = PreTrainedTokenizerFast.from_pretrained("gogamza/kobart-base-v2", model_max_length = 512)