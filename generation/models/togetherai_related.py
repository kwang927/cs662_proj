import os
from together import Together


together_ai_key = "replace_with_your_togetherai_api_key"

def togetherai_generation(prompt,model,temperature=0,max_tokens=1024,top_p=0.95,frequency_penalty=0,presence_penalty=0):
    client = Together(api_key=together_ai_key)

    conversation = [{'role': 'user', 'content': prompt}]

    response = client.chat.completions.create(
    model=model,
    messages=conversation,
    temperature=temperature,
    max_tokens=max_tokens,
    top_p=top_p,
    frequency_penalty=frequency_penalty,
    presence_penalty=presence_penalty)

    return response.choices[0].message.content, (response.usage.prompt_tokens, response.usage.completion_tokens)

def togetherai_generation_multi_turn(prompt,model,temperature=0,max_tokens=1024,top_p=0.95,frequency_penalty=0,presence_penalty=0):


    if type(prompt) is not dict or 'conversation' not in prompt.keys():
        raise ValueError("For multiple turn case, the prompt dict provided is not in correct format")
    
    client = Together(api_key="together_ai_key")

    conversation = prompt['conversation']

    response = client.chat.completions.create(
        model=model,
        messages=conversation,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty)

    return response.choices[0].message.content, (response.usage.prompt_tokens, response.usage.completion_tokens)

