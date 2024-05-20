import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

from huggingface_hub.hf_api import HfFolder
HfFolder.save_token('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')

tokenizer = AutoTokenizer.from_pretrained("arya123321/arya",
                                          token= "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",)

print("before auto causal")
model = AutoModelForCausalLM.from_pretrained("arya123321/arya",
                                             device_map='auto',
                                             
                                             torch_dtype=torch.float16,
                                             token="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",
                                             offload_folder  = "./offload"
                                            #  load_in_8bit=True,
                                            #load_in_4bit=True
                                             )
print("After auto causal")

import json
import textwrap

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """Generate a cooking recipe using the given ingredients.List all ingredients and steps in detail."""

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
    print("after get_prompt()")
    with torch.autocast('cpu', dtype=torch.bfloat16):
        inputs = tokenizer(prompt, return_tensors="pt")#.to('cuda')
        print("after inputs")
        outputs = model.generate(**inputs,
                                 max_new_tokens=512,
                                 eos_token_id=tokenizer.eos_token_id,
                                 pad_token_id=tokenizer.eos_token_id,
                                 )
        print("After generate")
        final_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        final_outputs = cut_off_text(final_outputs, '</s>')
        final_outputs = remove_substring(final_outputs, prompt)
    print("Reached end")

    return final_outputs#, outputs

def parse_text(text):
        wrapped_text = textwrap.fill(text, width=100)
        print(wrapped_text +'\n\n')
        # return assistant_text

prompt = 'bread,egg,milk'
print("Before generate()")
generated_text = generate(prompt)
print(f"geerated text: {generated_text}")
parse_text(generated_text)
