from xai_sdk import Client
from xai_sdk.chat import user, system

xai_api_key = "replace_with_your_xai_api_key"

def generate_grok_response_multi_turn(prompt, model="grok-4-0709", temperature=0, max_tokens = 1024, top_p=0.95,frequency_penalty=0,presence_penalty=0):
    # Note: max_tokens, top_p, frequency_penalty, and presence_penalty are not used in the current implementation of xai_sdk
    client = Client(api_host="api.x.ai", api_key=xai_api_key)

    chat = client.chat.create(model=model, temperature=temperature)
    if type(prompt) is not dict or 'conversation' not in prompt.keys():
        raise ValueError("For multiple turn case, the prompt dict provided is not in correct format")
    
    for turn in prompt['conversation']:
        if turn['role'] == 'user':
            chat.append(user(turn['content']))
        elif turn['role'] == 'assistant':
            chat.append(system(turn['content']))
        elif turn['role'] == 'system':
            chat.append(system(turn['content']))
        else:
            raise ValueError("Invalid role in conversation. Expected 'user' or 'assistant'.")
    
    response = chat.sample()

    return response.content, (response.usage.prompt_tokens, response.usage.reasoning_tokens)

