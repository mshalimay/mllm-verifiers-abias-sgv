# fmt:off

import torch
from PIL import Image
from transformers import Blip2ForConditionalGeneration, Blip2Processor

torch.cuda.empty_cache()

device = "cuda"
dtype = "bfloat16"

processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl", force_download=True)
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl", torch_dtype=dtype)


# processor.num_query_tokens = model.config.num_query_tokens
# image_token = AddedToken("<image>", normalized=False, special=True)
# processor.tokenizer.add_tokens([image_token], special_tokens=True)

# model.resize_token_embeddings(len(processor.tokenizer), pad_to_multiple_of=64)  # pad for efficient computation
# model.config.image_token_index = len(processor.tokenizer) - 1

device = "cuda"
model.to(device)


# Open the image using PIL
# image = Image.open("scripts/utils/image.png").convert("RGB")
image = Image.open("B075R7BFV2.0.jpg").convert("RGB")
images = [image] * 1
prompt = [""] * len(images)
# Process the image
inputs = processor(images=images, text=prompt, return_tensors="pt").to(device)

# Generate caption
generated_ids = model.generate(**inputs, max_new_tokens=30)
caption = processor.batch_decode(generated_ids, skip_special_tokens=True)
print(caption)


# Save the model
save_path = "~/.cache/huggingface/hub/models--Salesforce--blip2-flan-t5-xl"
model.save_pretrained(save_path)
processor.save_pretrained(save_path)
