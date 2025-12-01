import pandas as pd
import google.generativeai as genai
from typing import List, Optional
from .._constants import PRIMARY_GOOGLE_API_KEY, EMBEDDING_MODEL_NAME

def get_embedding(text: str, task_type: str, model: str = EMBEDDING_MODEL_NAME, api_keys: Optional[List[str]] = None, max_wait_time: int = 120, max_consecutive_failures_per_key: int = 3, max_total_attempts: int = 15):
    if not isinstance(text, str) or not text.strip() or pd.isna(text):
        return None
    if api_keys is None or not api_keys:
        api_keys = [PRIMARY_GOOGLE_API_KEY]
    text = text[:8000]
    total_attempts = 0
    current_key_index = 0
    consecutive_failures_with_current_key = 0
    current_wait_time = 5
    while total_attempts < max_total_attempts:
        current_api_key = api_keys[current_key_index]
        total_attempts += 1
        try:
            genai.configure(api_key=current_api_key)
            result = genai.embed_content(model=model, content=text, task_type=task_type)
            return result["embedding"]
        except Exception:
            consecutive_failures_with_current_key += 1
            if consecutive_failures_with_current_key >= max_consecutive_failures_per_key:
                current_key_index = (current_key_index + 1) % len(api_keys)
                consecutive_failures_with_current_key = 0
            current_wait_time = min(current_wait_time * 2, max_wait_time)
    return None

