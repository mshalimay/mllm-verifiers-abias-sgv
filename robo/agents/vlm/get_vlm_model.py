from agents.vlm.open_ai_vlm import OpenAIVLM

def get_vlm_model(vlm_name:str="gpt-4o",temperature_set=0,max_tokens=3000):
    if "gpt-4o" in vlm_name:
        return OpenAIVLM(model_name="gpt-4o", temperature_set=temperature_set, max_tokens=max_tokens)
    else:
        return None