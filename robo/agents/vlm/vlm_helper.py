import os
import base64
from PIL import Image

def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text


def get_image_list(image_folder):
    image_list = []
    for img in os.listdir(image_folder):
        if img.endswith(".png"):
            image_list.append(f"{image_folder}/{img}")
    return image_list


def combine_images_side_by_side(image_paths, output_path):
    # Load all images
    images = [Image.open(path) for path in image_paths]

    # Find the total width and maximum height of the final image
    total_width = sum(image.width for image in images)
    max_height = max(image.height for image in images)

    # Create a new blank image with the total width and max height
    combined_image = Image.new("RGB", (total_width, max_height))

    # Paste each image next to each other
    x_offset = 0
    for img in images:
        combined_image.paste(img, (x_offset, 0))
        x_offset += img.width

    # Save the combined image
    combined_image.save(output_path)
    print(f"Combined image saved at {output_path}")
    return output_path


def encode_image_func(image_path):
    """
        Encodes an image for VLM query.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def format_prompt(text_prompt, image_list):
    """
    Encodes the conversation for querying the OpenAI api.
    """
    formatted_prompt = [{"type": "text", "text": text_prompt}]
    for img_path in image_list:
        if not os.path.exists(img_path):
            print(f"Error: Image doesn't exist at provided path: {img_path}")
            exit(1)

        base64_image = encode_image_func(img_path)
        formatted_prompt.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_image}"}
            }
        )

    return formatted_prompt




