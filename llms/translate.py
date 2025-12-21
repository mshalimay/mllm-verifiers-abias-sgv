import codecs
import difflib
import json

from ftfy import fix_text
from langdetect import detect
from pydantic import BaseModel

from core_utils.logger_utils import logger
from llms.llm_utils import call_llm

# from googletrans import Translator
# translator = Translator()

LLM_TRANSLATOR_MODEL = "gemini-2.0-flash-001"


def is_english(text):
    # Check 1: Only contains printable ASCII (letters, digits, punctuation, etc.)
    ascii_check = all(ord(c) < 128 for c in text)

    # Check 2: Language detection (optional for short texts, can be unreliable)
    try:
        lang_check = detect(text) == "en"
    except Exception as _:
        lang_check = False

    if ascii_check and lang_check:
        # print(f"Is English: {text}")
        return True
    else:
        return False


def unescape_unicode(s: str) -> str:
    """
    Turn literal escape sequences like '\\u6e05\\u7406' into real characters.
    """
    try:
        return codecs.decode(s, "unicode_escape")
    except Exception:
        return s


def fix_mojibake(s: str) -> str:
    """
    Try re-interpreting the string as bytes in Latin-1 or CP1252, then decoding as UTF-8.
    This fixes garbled text like 'æˆ‘çœ‹…'.
    """
    for enc in ("latin1", "cp1252"):
        try:
            candidate = s.encode(enc).decode("utf-8")
            # Heuristic: if we pick up any CJK (Chinese/Japanese/Korean) block characters,
            # that's a good sign we recovered text properly.
            if any("\u4e00" <= ch <= "\u9fff" for ch in candidate):
                return candidate
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
    return s


def normalise(text: str) -> str:
    """
    Accepts any of your `content='…'` strings and returns cleaned, readable text.
    """
    text = unescape_unicode(text)
    text = fix_text(text)

    # text = fix_mojibake(text)

    return text


class InputToEnglish(BaseModel):
    original: str
    translated: str


def parse_output(json_str: str, original_texts: list[str]) -> list[str] | None:
    """
    Parse the JSON output from the translation LLM, ensuring that the order is preserved
    by matching the similarity between the 'original' field in the output objects and the original input texts.

    Args:
        json_str (str): A JSON string representing a list of translation objects.
        original_texts (list[str]): The original list of texts that were translated.

    Returns:
        list[str] | None: A list of translated texts in the same order as original_texts,
        or None if parsing fails or if a sufficient matching cannot be found.
    """
    try:
        # Parse the JSON response, expecting a list of dicts with keys 'original' and 'translated'
        translations = json.loads(json_str)

        if not isinstance(translations, list):
            logger.error("Parsed JSON is not a list.")
            return None

        if len(translations) != len(original_texts):
            logger.error("Mismatch between number of original texts and translations.")
            return None

        ordered_translations = []
        # We'll mark items as used to ensure each translation is matched only once.
        used = [False] * len(translations)
        # Define a similarity threshold: adjust as needed (0.8 is a reasonable default).
        SIMILARITY_THRESHOLD = 0.8

        for input_text in original_texts:
            best_match_index = None
            best_similarity = 0.0

            # For each input text, find the translation object with the most similar original.
            for idx, trans_obj in enumerate(translations):
                if used[idx]:
                    continue
                # Get the original text returned by the LLM; default to "" if missing.
                returned_orig = trans_obj.get("original", "")
                similarity = difflib.SequenceMatcher(None, input_text, returned_orig).ratio()
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_index = idx

            if best_match_index is None or best_similarity < SIMILARITY_THRESHOLD:
                logger.error(f"Could not find a matching translation for input: {input_text}. Best similarity: {best_similarity}")
                return None

            # Mark this translation object as used.
            used[best_match_index] = True
            ordered_translations.append(translations[best_match_index].get("translated"))

        if len(ordered_translations) != len(original_texts):
            logger.error(f"Mismatch between number of original texts and translations: {len(ordered_translations)} != {len(original_texts)}")
            return None
        return ordered_translations

    except Exception as e:
        logger.error(f"Failed to parse JSON: {json_str} - Error: {e}")
        return None


def chunked(lst: list, n: int):
    """
    Yield successive n-sized chunks from the list.
    """
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def batch_to_english(
    texts: list[str],
    translate_backend: str = "llm",
    model: str = LLM_TRANSLATOR_MODEL,
    max_batch_size: int = 30,
) -> list[str] | None:
    try:
        sys_prompt = """
        You are a helpful assistant that excels at translating any type of textual input to English.

        For each input text provided, please output a JSON object with the following structure:
        {
            "original": "<the original input text>",
            "translated": "<the full English translation of the input text, ensuring that any non-English content is accurately translated>"
        }

        Return a JSON array containing such objects for all the input texts. Do not include any extra commentary or formatting.
        """.strip()

        # Normalize all texts.
        normalized_texts = [normalise(text) for text in texts]
        results = normalized_texts[:]  # shallow copy

        # Identify non-English texts and record their indices.
        non_english_texts = []
        non_english_indices = []
        for idx, text in enumerate(normalized_texts):
            if not is_english(text):
                non_english_texts.append(text)
                non_english_indices.append(idx)

        # If no translations are needed, return the normalized texts.
        if not non_english_texts:
            return results

        translations: list[str] = []
        if len(non_english_texts) == 1:
            # If there's just one non-English text, use the individual translation call.
            single_translation = to_english(non_english_texts[0], translate_backend=translate_backend, model=model)
            if single_translation is None:
                logger.error("Failed to translate text.")
                return None
            translations = [single_translation]
        else:
            # Process non-English texts in chunks defined by max_batch_size.
            for chunk in chunked(non_english_texts, max_batch_size):
                # Build the prompt for this chunk of texts.
                joined_text = "\n".join([f"{i + 1}. {text}" for i, text in enumerate(chunk)])
                msgs = [
                    {
                        "role": "system",
                        "content": sys_prompt,
                    },
                    {
                        "role": "user",
                        "content": f"Please translate the following texts to English:\n\n{joined_text}",
                    },
                ]
                gen_kwargs = {
                    "num_generations": 1,
                    "max_tokens": 8192,
                    "temperature": 0,
                    "model": model,
                    "engine": "automodel",
                    "response_schema": list[InputToEnglish],
                }

                _, model_messages = call_llm(gen_kwargs, msgs)
                chunk_translations = parse_output(model_messages[0].text(), chunk)
                if chunk_translations is None:
                    logger.error("Failed to parse translations from LLM response.")
                    return None
                translations.extend(chunk_translations)

        # Merge translations back into the results using the original indices.
        for orig_idx, translation in zip(non_english_indices, translations):
            results[orig_idx] = translation

        return results
    except Exception as e:
        logger.error(f"Failed to translate texts: {texts} - Error: {e}")
        return None
    # finally:
    #     restore_api_keys_to_file()


def to_english(text: str, translate_backend: str = "llm", model: str = "Qwen/Qwen3-8B") -> str:
    # fmt:off
    sys_prompt = """
    You are a helpful assistant that excels on translating any type of textual input to English.
    # To be succesful, it is very important that you follow the following rules:
    1. You should only respond with the translated text, nothing else.
    2. If parts of the text are in English, translate the remaining parts making sure the final text is coherent and the meaning is consistent.
    3. If the text is already in English, just return it.
    4. Always return the FULL counterpart of the original text.
    """


    normalized_text = normalise(text)

    if is_english(normalized_text):
        logger.info(f"Already in English: {normalized_text}")
        return normalized_text

    # fmt:on
    if translate_backend == "llm":
        msgs = [
            {
                "role": "system",
                "content": sys_prompt,
            },
            {
                "role": "user",
                "content": f"Please translate the below text to English:\n\n{normalized_text}",
            },
        ]
        gen_kwargs = {
            "num_generations": 1,
            "max_tokens": 8192,
            "temperature": 0,
            # "top_p": 0.001,
            # "top_k": 40,
            "model": model,
            "engine": "automodel",
            "device": "cpu",
        }

        print("Calling LLM...")
        api_response, model_generations = call_llm(gen_kwargs, msgs)
        return model_generations[0].text()

    elif translate_backend == "googletrans":
        #  return translator.translate(text, dest="en").text
        raise NotImplementedError("Googletrans is not implemented")
    else:
        raise ValueError(f"Unknown translate backend: {translate_backend}")
