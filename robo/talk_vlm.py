'''
This script is used to test the VLM model.
It is used to test the 1-pass and 2-pass methods.

NOTE: Need to run the unittest first to generate the figure and language files.
'''
from agents.vlm.get_vlm_model import get_vlm_model
from agents.vlm.vlm_helper import read_text_file, get_image_list, combine_images_side_by_side

if __name__ == "__main__":
    vlm = get_vlm_model(vlm_name="gpt-4o", temperature_set=0, max_tokens=3000)

    # 1-pass baseline prompt.
    text_prompt = read_text_file("instruction/tool_hang_1pass_prompt.txt") + read_text_file("instruction/tool_hang.txt")
    one_pass_response = vlm.get_response(text_prompt=text_prompt, image_list=["tests/figure/tool_hang/ph/mlp/rollout_001/125.png"])
    print(one_pass_response) 

    # 2-pass prompt sequence.
    text_prompt_part1 = read_text_file("instruction/tool_hang_2pass_prompt_part1.txt") + read_text_file("instruction/tool_hang.txt")
    prediction = vlm.get_response(text_prompt=text_prompt_part1, image_list=["tests/figure/tool_hang/ph/mlp/rollout_001/125.png"])
    text_prompt = read_text_file("instruction/tool_hang_2pass_prompt_part2.txt") + "\nPrediction:\n" + prediction
    two_pass_response = vlm.get_response(text_prompt=text_prompt, image_list=["tests/figure/tool_hang/ph/mlp/rollout_001/125.png"])
    
 
 