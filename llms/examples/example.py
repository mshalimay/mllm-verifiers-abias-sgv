from PIL import Image

from llms.llm_utils import call_llm

prompt = [
    {"role": "system", "text": "You are an intelligent and helpful assistant that talks like a pirate."},
    "Describe **all** the below items.",
    ["Item (1):", "examples/dog.png"],
    ["Item (2):", Image.open("examples/cat.png")],
    ["Item (3):", "Once upon a time, there was a princess who lived in a castle."],
    ["Provide your response as follows: <Title for Item 1> <Description for Item 1> <Title for Item 2> <Description for Item 2> <Title for Item 3> <Description for Item 3>"],
]

gen_args = {
    "model": "gemini-2.0-flash-lite",
    "num_generations": 1,
    "temperature": 0.6,
    "top_p": 0.9,
    "top_k": 40,
    "max_tokens": 8192,
    "include_logprobs": True,
}

api_response, model_generations = call_llm(gen_args, prompt)

print(api_response)
print(model_generations[0].thoughts())
print(model_generations[0].text())
