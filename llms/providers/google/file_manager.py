import os
import tempfile
from typing import Any, Dict

from google import genai
from google.genai import types as genai_types
from PIL import Image

from core_utils.image_utils import any_to_pil
from llms.providers.google.google_client_manager import get_client_manager


class GoogleFileManager:
    def __init__(self, p_id: int = 0):
        self.p_id = p_id
        # Maps images to genai files. Why: GoogleAPI requires uploading big images to the cloud
        # to send in the prompts. This dictionary maps images previously uploaded to the genai files.
        self.img_to_uploaded: Dict[int, genai_types.File] = {}  # hash(image) -> genai file

        # Maps genai files to PIL images. This necessary because it is not possible to retrieve the images back from genai files.
        # and is useful to create prompt visualizations, reupload images
        self.uploaded_to_img: Dict[str, Image.Image] = {}  # genai file -> image

    def upload_image_file(
        self,
        image_path: str,
        client: genai.Client,
    ) -> genai_types.File:
        return client.files.upload(file=image_path)

    def get_upload_image_file(self, image: Any, force_upload: bool = False) -> genai_types.File:
        image_pil = any_to_pil(image)
        image_hash = hash(image_pil.tobytes())

        if image_hash in self.img_to_uploaded and not force_upload:
            gen_ai_file = self.img_to_uploaded[image_hash]
            if gen_ai_file.uri is None:
                raise ValueError("Uploaded file has no valid URI")
            self.uploaded_to_img[gen_ai_file.uri] = image_pil
            return gen_ai_file

        # If the image is not in the cache or force_upload is True, upload the image to the cloud.
        client = get_client_manager(self.p_id).get_client()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp:
            image_pil.save(temp.name, format="PNG")
            gen_ai_file = self.upload_image_file(temp.name, client)
            if gen_ai_file.uri is None:
                raise ValueError("Uploaded file has no valid URI")
            self.uploaded_to_img[gen_ai_file.uri] = image_pil
            self.img_to_uploaded[image_hash] = gen_ai_file
        os.remove(temp.name)
        return self.img_to_uploaded[image_hash]

    def reupload_image(self, gen_ai_filename: str, pop_from_cache: bool = True) -> genai_types.File:
        if pop_from_cache:
            original_img = self.uploaded_to_img.pop(gen_ai_filename)
        else:
            original_img = self.uploaded_to_img[gen_ai_filename]

        new_gen_ai_file = self.get_upload_image_file(image=original_img, force_upload=True)
        return new_gen_ai_file

    def get_files_from_prompt(self, prompt: list[genai_types.Content]) -> list[str]:
        files = []
        for i, content in enumerate(prompt):
            if not content.parts:
                continue
            for j, part in enumerate(content.parts):
                if hasattr(part, "file_data") and part.file_data is not None:
                    files.append(part.file_data.file_uri)
        return files

    def reupload_images_for_prompt(self, prompt) -> Dict[str, genai_types.File]:
        unique_files = list(dict.fromkeys(self.get_files_from_prompt(prompt)))
        old_to_new: Dict[str, genai_types.File] = {}
        for old_file in unique_files:
            if old_file in self.uploaded_to_img:
                new_gen_ai_file = self.reupload_image(old_file)
            else:
                img = self.uploaded_to_img.get(old_file)
                if img is None:
                    raise ValueError(f"File {old_file} not found in cache")
                new_gen_ai_file = self.get_upload_image_file(image=img, force_upload=True)
            old_to_new[old_file] = new_gen_ai_file
        return old_to_new

    def reupload_all_images(self, keep_list: list[genai_types.File] = []) -> Dict[str, genai_types.File]:
        if keep_list:
            keys_to_keep = set([file.uri for file in keep_list if hasattr(file, "uri")])
        else:
            keys_to_keep = set()

        old_gen_ai_files = self.uploaded_to_img.keys() - keys_to_keep
        return {old_file: self.reupload_image(old_file) for old_file in old_gen_ai_files}

    def flush_cache(self, keep_list: list[genai_types.File] = []) -> None:
        if keep_list:
            keys_to_keep = set([file.uri for file in keep_list if hasattr(file, "uri") and file.uri is not None])
        else:
            keys_to_keep = set()
        self.uploaded_to_img = {k: v for k, v in self.uploaded_to_img.items() if k in keys_to_keep}
        self.img_to_uploaded = {k: v for k, v in self.img_to_uploaded.items() if v.uri in keys_to_keep and v.uri is not None}


google_file_managers = {}


def get_file_manager(p_id: int = 0) -> GoogleFileManager:
    if p_id not in google_file_managers:
        google_file_managers[p_id] = GoogleFileManager(p_id=p_id)
    return google_file_managers[p_id]
