import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

# notebook_login()
from huggingface_hub.hf_api import HfFolder
HfFolder.save_token('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')

tokenizer = AutoTokenizer.from_pretrained("arya123321/arya",
                                          token= "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",)

model = AutoModelForCausalLM.from_pretrained("arya123321/arya",
                                             device_map='cpu',
                                             torch_dtype=torch.float16,
                                             token="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",
                                            #  load_in_8bit=True,
                                            #load_in_4bit=True
                                             )

from transformers import pipeline

import json
import textwrap

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """\Generate a cooking recipe using the given ingredients.List all ingredients and steps in detail."""

SYSTEM_PROMPT = B_SYS + DEFAULT_SYSTEM_PROMPT + E_SYS


def get_prompt(instruction):
    prompt_template =  B_INST + SYSTEM_PROMPT + instruction + E_INST
    return prompt_template

def cut_off_text(text, prompt):
    cutoff_phrase = prompt
    index = text.find(cutoff_phrase)
    if index != -1:
        return text[:index]
    else:
        return text

def remove_substring(string, substring):
    return string.replace(substring, "")



def generate(text):
    prompt = get_prompt(text)
    with torch.autocast('cpu', dtype=torch.bfloat16):
      
        outputs = model.generate(**inputs,
                                 max_new_tokens=512,
                                 eos_token_id=tokenizer.eos_token_id,
                                 pad_token_id=tokenizer.eos_token_id,
                                 )
        final_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        final_outputs = cut_off_text(final_outputs, '</s>')
        final_outputs = remove_substring(final_outputs, prompt)

    return final_outputs#, outputs

def parse_text(text):
        wrapped_text = textwrap.fill(text, width=100)
        print(wrapped_text +'\n\n')
        # return assistant_text
