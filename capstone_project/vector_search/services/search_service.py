import time
from typing import List, Dict
import pandas as pd
from tqdm import tqdm
import requests
import json
from concurrent.futures import ThreadPoolExecutor
from .._constants import (
    DETAIL_COLUMN_NAME,
    MAX_RETRIES_LLM,
    INITIAL_RETRY_DELAY_LLM,
    BATCH_SIZE_LLM,
    MAX_CONCURRENT_REQUESTS_LLM,
    GEMINI_API_KEYS,
    BATCH_GROUP_DELAY_LLM,
)
from .embedding_service import get_embedding

def embed_and_upsert_profiles(df: pd.DataFrame, collection, batch_size_chroma: int = 100) -> None:
    profiles_to_upsert: List[Dict] = []
    processed_count = 0
    failed_count = 0
    total_profiles = len(df)
    progress_bar = tqdm(total=total_profiles, desc="Embedding & Upserting")
    for index, row in df.iterrows():
        text_to_embed = row.get(DETAIL_COLUMN_NAME)
        if pd.isna(text_to_embed) or not isinstance(text_to_embed, str) or not text_to_embed.strip():
            progress_bar.update(1)
            continue
        embedding = get_embedding(text_to_embed, task_type="RETRIEVAL_DOCUMENT")
        if embedding:
            metadata: Dict = {}
            for col in ['Tiêu đề', 'Họ và tên', 'Link', 'Năm sinh', 'Năm thất lạc']:
                if col in row:
                    value = row[col]
                    if pd.isna(value):
                        metadata[col] = ""
                    elif isinstance(value, (str, int, float, bool)):
                        metadata[col] = value
                    else:
                        metadata[col] = str(value)
            metadata = {k: "" if pd.isna(v) else v for k, v in metadata.items()}
            profiles_to_upsert.append({"id": str(index), "embedding": embedding, "metadata": metadata})
            if len(profiles_to_upsert) >= batch_size_chroma:
                try:
                    collection.upsert(
                        ids=[p["id"] for p in profiles_to_upsert],
                        embeddings=[p["embedding"] for p in profiles_to_upsert],
                        metadatas=[p["metadata"] for p in profiles_to_upsert],
                    )
                    processed_count += len(profiles_to_upsert)
                    profiles_to_upsert = []
                except Exception:
                    failed_count += len(profiles_to_upsert)
                    profiles_to_upsert = []
        else:
            failed_count += 1
        progress_bar.update(1)
    if profiles_to_upsert:
        try:
            collection.upsert(
                ids=[p["id"] for p in profiles_to_upsert],
                embeddings=[p["embedding"] for p in profiles_to_upsert],
                metadatas=[p["metadata"] for p in profiles_to_upsert],
            )
            processed_count += len(profiles_to_upsert)
        except Exception:
            failed_count += len(profiles_to_upsert)
    progress_bar.close()

def verify_profiles_with_llm(query: str, profiles_data: List[dict], api_key: str) -> List[str]:
    profile_strings: List[str] = []
    for profile in profiles_data:
        profile_id = profile.get('id') if isinstance(profile, dict) else profile.name
        title = profile.get('Tiêu đề', 'N/A')
        name = profile.get('Họ và tên', 'N/A')
        detail_source = profile.get('metadata', {}) if isinstance(profile, dict) and 'metadata' in profile else profile
        detail = detail_source.get(DETAIL_COLUMN_NAME, 'N/A')
        detail = str(detail).replace('\\', '/')[:1000]
        profile_strings.append(
            f"\nIndex: {profile_id}\nTiêu đề: {title}\nHọ tên: {name}\nChi tiết: {detail}\n{'-'*40}"
        )
    prompt = f"""Bạn là một chuyên gia phân tích hồ sơ tìm kiếm người thân thất lạc cực kỳ tỉ mỉ và chính xác. Nhiệm vụ của bạn là phân tích các hồ sơ dưới đây và chỉ xác định những hồ sơ nào mô tả **chính xác cùng một người** và **cùng một hoàn cảnh thất lạc** như được nêu trong Yêu cầu tìm kiếm.\n\nHãy so sánh **cực kỳ cẩn thận** các chi tiết nhận dạng cốt lõi:\n- **Họ tên người thất lạc:** Phải khớp hoặc rất tương đồng.\n- **Tên cha, mẹ, anh chị em (nếu có trong yêu cầu):** Phải khớp hoặc rất tương đồng.\n- **Năm sinh:** Phải khớp hoặc gần đúng.\n- **Quê quán/Địa chỉ liên quan:** Phải khớp hoặc có liên quan logic.\n- **Hoàn cảnh thất lạc (thời gian, địa điểm, sự kiện chính):** Phải tương đồng đáng kể.\n\n**Quy tắc loại trừ quan trọng:**\n- Nếu **Họ tên người thất lạc** trong hồ sơ **khác biệt rõ ràng** so với yêu cầu, hãy **LOẠI BỎ** hồ sơ đó NGAY LẬP TỨC, bất kể các chi tiết khác có trùng khớp hay không.\n- Nếu **tên cha mẹ hoặc anh chị em** (khi được cung cấp trong yêu cầu) trong hồ sơ **hoàn toàn khác biệt**, hồ sơ đó rất có thể **KHÔNG PHÙ HỢP** và cần được xem xét loại bỏ.\n- Sự trùng khớp **chỉ** về địa danh hoặc năm sinh là **KHÔNG ĐỦ** để kết luận hồ sơ phù hợp nếu các tên riêng cốt lõi và hoàn cảnh thất lạc khác biệt.\n\nMỗi hồ sơ có Index gốc, Tiêu đề, Họ tên và Chi tiết mô tả.\n\nYêu cầu tìm kiếm:\n{query}\n------------------------------------\n\nCác hồ sơ cần kiểm tra:\n------------------------------------\n{"".join(profile_strings)}\n------------------------------------\n\nHãy trả về **chỉ các Index gốc** (là các chuỗi ID dạng số) của những hồ sơ mà bạn **rất chắc chắn** (high confidence) là phù hợp dựa trên **tất cả các tiêu chí cốt lõi** nêu trên. Mỗi index trên một dòng. Nếu không có hồ sơ nào thực sự phù hợp, trả về 'none'."""
    api_endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 256},
        "safetySettings": [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ],
    }
    for attempt in range(MAX_RETRIES_LLM):
        try:
            response = requests.post(api_endpoint, headers=headers, json=payload, timeout=60)
            if response.status_code in (429,) or response.status_code >= 500:
                if attempt < MAX_RETRIES_LLM - 1:
                    wait_time = INITIAL_RETRY_DELAY_LLM * (2 ** attempt)
                    time.sleep(wait_time)
                    continue
                return []
            if response.status_code != 200:
                try:
                    error_json = response.json().get('error', {})
                    error_detail = error_json.get('message', response.text)
                    if "API key not valid" in error_detail:
                        return []
                except json.JSONDecodeError:
                    pass
                return []
            response_data = response.json()
            try:
                if response_data.get('promptFeedback', {}).get('blockReason'):
                    return []
                generated_text = response_data['candidates'][0]['content']['parts'][0]['text']
                if generated_text:
                    if generated_text.strip().lower() == 'none':
                        return []
                    matched_indices_str = [idx.strip() for idx in generated_text.split('\n') if idx.strip().isdigit()]
                    return matched_indices_str
                return []
            except (KeyError, IndexError, TypeError):
                return []
        except requests.exceptions.RequestException:
            if attempt < MAX_RETRIES_LLM - 1:
                wait_time = INITIAL_RETRY_DELAY_LLM * (2 ** attempt)
                time.sleep(wait_time)
            else:
                return []
    return []

def parallel_verify(query: str, ranked_profiles_data: List[dict], max_profiles: int = 300) -> List[str]:
    max_profiles = min(max_profiles, len(ranked_profiles_data))
    profiles_to_verify = ranked_profiles_data[:max_profiles]
    if not profiles_to_verify:
        return []
    batches = [profiles_to_verify[i:i + BATCH_SIZE_LLM] for i in range(0, len(profiles_to_verify), BATCH_SIZE_LLM)]
    verified_indices_str = set()
    num_api_keys = len(GEMINI_API_KEYS)
    batch_groups = [batches[i:i + MAX_CONCURRENT_REQUESTS_LLM] for i in range(0, len(batches), MAX_CONCURRENT_REQUESTS_LLM)]
    for batch_group in batch_groups:
        with ThreadPoolExecutor(max_workers=min(len(batch_group), MAX_CONCURRENT_REQUESTS_LLM)) as executor:
            futures = {}
            for i, batch in enumerate(batch_group):
                if not batch:
                    continue
                api_key_index = i % num_api_keys
                api_key = GEMINI_API_KEYS[api_key_index]
                future = executor.submit(verify_profiles_with_llm, query, batch, api_key)
                futures[future] = i + 1
            for future in futures:
                try:
                    result = future.result()
                    if result:
                        verified_indices_str.update(result)
                except Exception:
                    pass
        time.sleep(BATCH_GROUP_DELAY_LLM)
    return list(verified_indices_str)

def extract_keywords_gemini(query: str, model: str = "gemini-2.0-flash") -> List[str]:
    import google.generativeai as genai
    try:
        response = genai.GenerativeModel(model).generate_content(query)
        if response.text:
            keywords_str = response.text.strip()
            keywords = [kw.strip() for kw in keywords_str.split(',') + keywords_str.split('\n') if kw.strip()]
            return keywords
        return []
    except Exception:
        return []

def search_combined_chroma(df_original: pd.DataFrame, collection, user_query: str, top_n_final: int = 100):
    keywords = extract_keywords_gemini(user_query)
    keyword_match_indices = set()
    keyword_match_counts: Dict[int, int] = {}
    if keywords:
        for keyword in keywords:
            for col in df_original.columns:
                if df_original[col].dtype == object:
                    try:
                        matches = df_original[col].str.contains(keyword, case=False, na=False)
                        matched_indices = df_original[matches].index
                        keyword_match_indices.update(matched_indices)
                        for idx in matched_indices:
                            keyword_match_counts[idx] = keyword_match_counts.get(idx, 0) + 1
                    except Exception:
                        continue
    valid_keyword_indices = [idx for idx in keyword_match_indices if idx in df_original.index]
    if not valid_keyword_indices:
        keyword_match_counts = {idx: 0 for idx in df_original.index}
    query_embedding = get_embedding(user_query, task_type="RETRIEVAL_QUERY")
    if query_embedding is None:
        ranked_by_keywords = sorted(keyword_match_counts.items(), key=lambda x: x[1], reverse=True)
        top_keyword_profiles = ranked_by_keywords[:top_n_final]
        profiles_for_llm = []
        for idx, _ in top_keyword_profiles:
            try:
                profile_data = df_original.loc[idx].copy()
                profile_data['id'] = str(idx)
                profiles_for_llm.append(profile_data)
            except KeyError:
                continue
        return parallel_verify(user_query, profiles_for_llm, max_profiles=len(profiles_for_llm))
    try:
        vector_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=collection.count(),
            include=['metadatas', 'distances'],
        )
    except Exception:
        return None
    vector_distances: Dict[int, float] = {}
    if vector_results and vector_results.get('ids') and vector_results['ids'][0]:
        vector_ids = vector_results['ids'][0]
        distances = vector_results['distances'][0]
        for i, id_str in enumerate(vector_ids):
            try:
                idx = int(id_str)
                vector_distances[idx] = 1 - distances[i]
            except ValueError:
                continue
    else:
        return None
    combined_scores: Dict[int, float] = {}
    KEYWORD_BONUS = 0.05
    for idx in df_original.index:
        vector_score = vector_distances.get(idx, 0)
        keyword_count = keyword_match_counts.get(idx, 0)
        total_score = vector_score + keyword_count * KEYWORD_BONUS
        if total_score > 0:
            combined_scores[idx] = total_score
    if not combined_scores:
        return None
    ranked_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    top_results = ranked_results[:top_n_final]
    profiles_for_llm = []
    for idx, _ in top_results:
        try:
            profile_data = df_original.loc[idx].copy()
            profile_data['id'] = str(idx)
            profiles_for_llm.append(profile_data)
        except KeyError:
            continue
    if not profiles_for_llm:
        return None
    return parallel_verify(user_query, profiles_for_llm, max_profiles=len(profiles_for_llm))

