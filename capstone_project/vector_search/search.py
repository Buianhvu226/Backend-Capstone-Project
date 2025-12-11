import numpy as np
import pandas as pd
import time
import math

from .config import DETAIL_COLUMN_NAME, PINECONE_TOP_K
from .embedding import get_embedding
from .llm_utils import extract_keywords_gemini, parallel_verify
from .qdrant_helper import get_qdrant_client, get_qdrant_collection

def search_combined_chroma(df_original, collection, user_query, top_n_final=100, return_json=False, user=None):
    """
    Thực hiện tìm kiếm kết hợp:
    1. Tìm tất cả hồ sơ có ít nhất 1 từ khóa trùng khớp
    2. Thực hiện vector search trên tất cả hồ sơ
    3. Tính tổng điểm = điểm tương đồng vector + (số từ khóa khớp × 0.05)
    4. Chọn top_n_final hồ sơ có tổng điểm cao nhất để LLM lọc tiếp
    """
    print("\n--- Bắt đầu Tìm kiếm (Kết hợp Từ Khóa và Vector Search -> LLM) ---")

    # --- Bước 1: Trích xuất từ khóa và tìm tất cả hồ sơ có ít nhất 1 từ khóa trùng khớp ---
    keywords = extract_keywords_gemini(user_query)
    print("Từ khóa trích xuất từ Gemini:", keywords)
    
    # Chỉ tạo thông báo nếu có user
    if user:
        from notifications.utils import create_notification
        create_notification(
            user=user,
            notification_type='profile_creating',
            content=f'Đang trích xuất các từ khóa từ Gemini: {keywords}',
            additional_data={
                'text': f'Đang trích xuất các từ khóa từ Gemini: {keywords}',
            }
        )

    # Tìm kiếm các hồ sơ chứa ít nhất một từ khóa
    keyword_match_indices = set()
    keyword_match_counts = {}

    if keywords:
        print("Đang tìm kiếm hồ sơ chứa từ khóa...")
        from notifications.utils import create_notification
        create_notification(
            user=user,
            notification_type='profile_creating',
            content=f'Đang tìm kiếm các hồ sơ chứa ít nhất 1 trong các từ khóa: {keywords}',
            additional_data={
                'text': f'Đang tìm kiếm các hồ sơ chứa ít nhất 1 trong các từ khóa: {keywords}',
            }
        )
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
        print("\nKhông tìm thấy hồ sơ nào khớp với từ khóa. Sẽ tìm kiếm bằng vector search trên toàn bộ dữ liệu.")
        from notifications.utils import create_notification
        create_notification(
            user=user,
            notification_type='profile_creating',
            content=f'Không tìm thấy hồ sơ nào khớp với từ khóa: {keywords}. Sẽ tìm kiếm bằng vector search trên toàn bộ dữ liệu.',
            additional_data={
                'text': f'Không tìm thấy hồ sơ nào khớp với từ khóa: {keywords}. Sẽ tìm kiếm bằng vector search trên toàn bộ dữ liệu.',
            }
        )
        keyword_match_counts = {idx: 0 for idx in df_original.index}
    else:
        print(f"\nTìm thấy {len(valid_keyword_indices)} hồ sơ khớp với ít nhất một từ khóa.")
        from notifications.utils import create_notification
        create_notification(
            user=user,
            notification_type='profile_creating',
            content=f'Tìm thấy {len(valid_keyword_indices)} hồ sơ chứa ít nhất 1 trong các từ khóa: {keywords}.',
            additional_data={
                'text': f'Tìm thấy {len(valid_keyword_indices)} hồ sơ chứa ít nhất 1 trong các từ khóa: {keywords}.',
            }
        )
        keyword_counts = list(keyword_match_counts.values())
        print(f"Thống kê số lượng từ khóa khớp: Min={min(keyword_counts)}, Max={max(keyword_counts)}, Avg={sum(keyword_counts)/len(keyword_counts):.2f}")

    # --- Bước 2: Thực hiện vector search trên toàn bộ collection ---
    print("\nĐang tạo embedding cho truy vấn...")
    from notifications.utils import create_notification
    create_notification(
        user=user,
        notification_type='profile_creating',
        content=f'Đang tạo mã hóa cho truy vấn: {user_query}',
        additional_data={
            'text': f'Đang tạo mã hóa cho truy vấn: {user_query}',
        }
    )   
    query_embedding = get_embedding(user_query, task_type="RETRIEVAL_QUERY")

    if query_embedding is None:
        print("Lỗi: Không thể tạo embedding cho truy vấn. Sử dụng chỉ kết quả từ khóa.")

        if not valid_keyword_indices:
            print("Không tìm thấy kết quả từ khóa và không thể thực hiện vector search. Kết thúc tìm kiếm.")
            from notifications.utils import create_notification
            create_notification(
                user=user,
                notification_type='profile_creating',
                content=f'Không tìm thấy kết quả từ khóa và không thể thực hiện vector search. Kết thúc tìm kiếm.',
                additional_data={
                    'text': f'Không tìm thấy kết quả từ khóa và không thể thực hiện vector search. Kết thúc tìm kiếm.',
                }
            )
            return None

        # Nếu không thể tạo embedding, xếp hạng chỉ dựa trên số lượng từ khóa khớp
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

        # Xác minh bằng LLM
        print(f"\nĐang xác minh {len(profiles_for_llm)} kết quả với Gemini LLM...")
        from notifications.utils import create_notification
        create_notification(
            user=user,
            notification_type='profile_creating',
            content=f'Đang xác minh {len(profiles_for_llm)} kết quả với Gemini LLM...',
            additional_data={
                'text': f'Đang xác minh {len(profiles_for_llm)} kết quả với Gemini LLM...',
            }
        )
        verified_indices_str = parallel_verify(user_query, profiles_for_llm, max_profiles=len(profiles_for_llm))

        # Hiển thị kết quả cuối
        if verified_indices_str:
            print(f"\n=== {len(verified_indices_str)} KẾT QUẢ PHÙ HỢP NHẤT SAU KHI LỌC BẰNG LLM ===")
            from notifications.utils import create_notification
            create_notification(
                user=user,
                notification_type='profile_creating',
                content=f'{len(verified_indices_str) - 1} kết quả phù hợp nhất sau khi lọc bằng LLM.',
                additional_data={
                    'text': f'{len(verified_indices_str) - 1} kết quả phù hợp nhất sau khi lọc bằng LLM.',
                }
            )
            for id_str in verified_indices_str:
                try:
                    idx = int(id_str)
                    profile = df_original.loc[idx]
                    print(f"\nIndex: {idx}")
                    print(f"Số từ khóa khớp: {keyword_match_counts.get(idx, 0)}")
                    print(f"Tiêu đề: {profile.get('Tiêu đề', 'N/A')}")
                    print(f"Họ và tên: {profile.get('Họ và tên', 'N/A')}")
                    print(f"Chi tiết: {str(profile.get(DETAIL_COLUMN_NAME, 'N/A'))[:500]}...")
                    print(f"Link: {profile.get('Link', 'N/A')}")
                    print("-" * 60)
                except Exception as e:
                    print(f"Lỗi khi hiển thị hồ sơ {id_str}: {e}")
        else:
            print("\nKhông tìm thấy hồ sơ nào phù hợp sau khi xác minh bằng LLM.")

        return verified_indices_str

    # Thực hiện vector search
    print("Đang thực hiện vector search trên toàn bộ collection...")
    from notifications.utils import create_notification
    create_notification(
        user=user,
        notification_type='profile_creating',
        content=f'Đang thực hiện vector search trên toàn bộ collection...',
        additional_data={
            'text': f'Đang thực hiện vector search trên toàn bộ collection...',
        }
    )
    try:
        vector_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=collection.count(),
            include=['metadatas', 'distances']
        )
    except Exception as e:
        print(f"Lỗi khi truy vấn ChromaDB: {e}")
        return None

    # --- Bước 3: Kết hợp kết quả từ khóa và vector ---
    print("Đang kết hợp kết quả từ khóa và vector...")
    from notifications.utils import create_notification
    create_notification(
        user=user,
        notification_type='profile_creating',
        content=f'Đang kết hợp kết quả từ khóa và vector...',
        additional_data={
            'text': f'Đang kết hợp kết quả từ khóa và vector...',
        }
    )

    vector_distances = {}
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
        print("Không nhận được kết quả từ vector search.")
        return None

    combined_scores = {}
    KEYWORD_BONUS = 0.05

    for idx in df_original.index:
        vector_score = vector_distances.get(idx, 0)
        keyword_count = keyword_match_counts.get(idx, 0)
        total_score = vector_score + keyword_count * KEYWORD_BONUS
        if total_score > 0:
            combined_scores[idx] = total_score

    if not combined_scores:
        print("Không tìm thấy hồ sơ nào có điểm kết hợp > 0.")
        return None

    print(f"Đang xếp hạng {len(combined_scores)} hồ sơ theo điểm kết hợp...")
    ranked_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    top_results = ranked_results[:top_n_final]

    print(f"\n--- Top {min(10, len(top_results))} Kết quả (Theo Điểm Kết Hợp, Trước LLM) ---")
    from notifications.utils import create_notification
    create_notification(
        user=user,  
        notification_type='profile_creating',
        content=f'Đã tìm thấy {len(top_results)} hồ sơ phù hợp sau khi kết hợp từ khóa và vector search.',
        additional_data={
            'text': f'Đã tìm thấy {len(top_results)} hồ sơ phù hợp sau khi kết hợp từ khóa và vector search.',
        }
    )
    for i, (idx, score) in enumerate(top_results[:10]):
        try:
            profile = df_original.loc[idx]
            kw_count = keyword_match_counts.get(idx, 0)
            kw_score = kw_count * KEYWORD_BONUS
            vec_score = vector_distances.get(idx, 0)
            print(f"  Rank: {i+1} | Tổng điểm: {score:.4f} (Vector: {vec_score:.4f}, KW: {kw_score:.4f}) | "
                  f"Từ khóa: {kw_count} | ID: {idx} | "
                  f"Tiêu đề: {profile.get('Tiêu đề', 'N/A')} | Họ và tên: {profile.get('Họ và tên', 'N/A')}")
            
            from notifications.utils import create_notification
            create_notification(
                user=user,
                notification_type='profile_creating',
                content=f'Hồ sơ {i+1} | ID: {idx} | Tiêu đề: {profile.get("Tiêu đề", "N/A")} | Họ và tên: {profile.get("Họ và tên", "N/A")}',
                additional_data={
                    'text': f'Hồ sơ {i+1} | ID: {idx} | Tiêu đề: {profile.get("Tiêu đề", "N/A")} | Họ và tên: {profile.get("Họ và tên", "N/A")}',
                }
            )
        except KeyError:
            print(f"  Rank: {i+1} | Tổng điểm: {score:.4f} | ID: {idx} | Lỗi: Không tìm thấy hồ sơ.")

    if len(top_results) > 10:
        print(f"  ... và {len(top_results) - 10} kết quả khác")

    profiles_for_llm = []
    for idx, _ in top_results:
        try:
            profile_data = df_original.loc[idx].copy()
            profile_data['id'] = str(idx)
            profiles_for_llm.append(profile_data)
        except KeyError:
            continue

    if not profiles_for_llm:
        print("Không tìm thấy hồ sơ hợp lệ để gửi đến LLM.")
        return None

    print(f"\nĐang xác minh {len(profiles_for_llm)} kết quả với Gemini LLM...")
    from notifications.utils import create_notification
    create_notification(
        user=user,
        notification_type='profile_creating',
        content=f'Đang xác minh {len(profiles_for_llm)} kết quả với Gemini LLM...',
        additional_data={
            'text': f'Đang xác minh {len(profiles_for_llm)} kết quả với Gemini LLM...',
        }
    )
    verified_indices_str = parallel_verify(user_query, profiles_for_llm, max_profiles=len(profiles_for_llm))

    if verified_indices_str:
        print(f"\n=== {len(verified_indices_str)} KẾT QUẢ PHÙ HỢP NHẤT SAU KHI LỌC BẰNG LLM ===")
        from notifications.utils import create_notification
        create_notification(
            user=user,
            notification_type='profile_creating',
            content=f'Đã tìm thấy {len(verified_indices_str) - 1} kết quả phù hợp nhất sau khi lọc bằng LLM.',
            additional_data={
                'text': f'Đã tìm thấy {len(verified_indices_str) - 1} kết quả phù hợp nhất sau khi lọc bằng LLM.',
            }
        )
        verified_indices_int = [int(id_str) for id_str in verified_indices_str if id_str.isdigit()]
        verified_with_scores = [(idx, combined_scores.get(idx, 0)) for idx in verified_indices_int if idx in combined_scores]
        verified_with_scores.sort(key=lambda x: x[1], reverse=True)

        for i, (idx, score) in enumerate(verified_with_scores):
            try:
                profile = df_original.loc[idx]
                kw_count = keyword_match_counts.get(idx, 0)
                kw_score = kw_count * KEYWORD_BONUS
                vec_score = vector_distances.get(idx, 0)
                print(f"\n{i+1}. ID: {idx}")
                print(f"Tổng điểm: {score:.4f} (Vector: {vec_score:.4f}, KW: {kw_score:.4f}, Từ khóa khớp: {kw_count})")
                print(f"Tiêu đề: {profile.get('Tiêu đề', 'N/A')}")
                print(f"Họ và tên: {profile.get('Họ và tên', 'N/A')}")
                print(f"Năm thất lạc: {profile.get('Năm thất lạc', 'N/A')}")
                print(f"Năm sinh: {profile.get('Năm sinh', 'N/A')}")
                print(f"Chi tiết: {str(profile.get(DETAIL_COLUMN_NAME, 'N/A'))[:500]}...")
                print("-" * 60)
            except Exception as e:
                print(f"Lỗi khi hiển thị hồ sơ {idx}: {e}")
    else:
        print("\nKhông tìm thấy hồ sơ nào phù hợp sau khi xác minh bằng LLM.")

    return verified_indices_str


def search_combined_pinecone(df_original, index, user_query, top_n_final=100, top_k=None, return_json=False, user=None):
    """
    Phiên bản dùng Pinecone thay cho Chroma.
    """
    top_k = top_k or PINECONE_TOP_K
    print("\n--- Bắt đầu Tìm kiếm (Pinecone + Từ Khóa -> LLM) ---")

    # B1: từ khóa (giữ nguyên logic)
    keywords = extract_keywords_gemini(user_query)
    keyword_match_indices = set()
    keyword_match_counts = {}
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

    # B2: embedding truy vấn
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

    # B3: query Pinecone
    try:
        res = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    except Exception as e:
        print(f"Lỗi khi truy vấn Pinecone: {e}")
        return None

    vector_distances = {}
    matches = getattr(res, "matches", None)
    if matches:
        for m in matches:
            try:
                idx = int(m.id)
                # Pinecone score thường là similarity (tùy metric). Dùng trực tiếp.
                vector_distances[idx] = m.score
            except Exception:
                continue
    else:
        print("Không nhận được kết quả từ Pinecone.")
        return None

    # B4: kết hợp điểm
    combined_scores = {}
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

    verified_indices_str = parallel_verify(user_query, profiles_for_llm, max_profiles=len(profiles_for_llm))

    if return_json and verified_indices_str:
        result_list = []
        for id_str in verified_indices_str:
            try:
                idx = int(id_str)
                profile = df_original.loc[idx]
                result_list.append({
                    "id": str(idx),
                    "total_score": combined_scores.get(idx, 0),
                    "vector_score": vector_distances.get(idx, 0),
                    "keyword_score": keyword_match_counts.get(idx, 0) * KEYWORD_BONUS,
                    "matched_keywords": keyword_match_counts.get(idx, 0),
                    "title": profile.get('Tiêu đề', ''),
                    "full_name": profile.get('Họ và tên', ''),
                    "losing_year": profile.get('Năm thất lạc', ''),
                    "born_year": profile.get('Năm sinh', ''),
                    "detail": str(profile.get(DETAIL_COLUMN_NAME, '')),
                    "link": profile.get('Link', ''),
                })
            except Exception:
                continue
        return result_list

    return verified_indices_str

    # Khi trả về kết quả cuối cùng:
    # After LLM filtering, suppose you have:
    # llm_final_indices = [list of indices after LLM filtering]
    # You also have all the score dicts: total_scores, vector_scores, keyword_scores, keyword_match_counts

    if return_json:
        result_list = []
        for idx in llm_final_indices:
            try:
                profile = df_original.loc[idx]
                result_list.append({
                    "id": str(idx),
                    "total_score": total_scores.get(idx, 0),
                    "vector_score": vector_scores.get(idx, 0),
                    "keyword_score": keyword_scores.get(idx, 0),
                    "matched_keywords": keyword_match_counts.get(idx, 0),
                    "title": profile.get('Tiêu đề', ''),
                    "full_name": profile.get('Họ và tên', ''),
                    "losing_year": profile.get('Năm thất lạc', ''),
                    "born_year": profile.get('Năm sinh', ''),
                    "detail": str(profile.get(DETAIL_COLUMN_NAME, '')),
                    "link": profile.get('Link', ''),
                })
            except Exception as e:
                print(f"Lỗi khi build kết quả JSON: {e}")
                continue
        return result_list
    else:
        print("\nKhông tìm thấy hồ sơ nào phù hợp sau khi xác minh bằng LLM.")

    return verified_indices_str


def search_combined_qdrant(df_original, qdrant_client, collection_name, user_query, top_n_final=100, return_json=False, user=None):
    """
    Thực hiện tìm kiếm kết hợp với Qdrant:
    1. Tìm tất cả hồ sơ có ít nhất 1 từ khóa trùng khớp
    2. Thực hiện vector search trên Qdrant
    3. Tính tổng điểm = điểm tương đồng vector + (số từ khóa khớp × 0.05)
    4. Chọn top_n_final hồ sơ có tổng điểm cao nhất để LLM lọc tiếp
    """
    print("\n--- Bắt đầu Tìm kiếm (Kết hợp Từ Khóa và Qdrant Vector Search -> LLM) ---")

    # --- Bước 1: Trích xuất từ khóa và tìm tất cả hồ sơ có ít nhất 1 từ khóa trùng khớp ---
    keywords = extract_keywords_gemini(user_query)
    print("Từ khóa trích xuất từ Gemini:", keywords)
    
    if user:
        from notifications.utils import create_notification
        create_notification(
            user=user,
            notification_type='profile_creating',
            content=f'Đang trích xuất các từ khóa từ Gemini: {keywords}',
            additional_data={'text': f'Đang trích xuất các từ khóa từ Gemini: {keywords}'}
        )

    keyword_match_indices = set()
    keyword_match_counts = {}

    if keywords:
        print("Đang tìm kiếm hồ sơ chứa từ khóa...")
        if user:
            from notifications.utils import create_notification
            create_notification(
                user=user,
                notification_type='profile_creating',
                content=f'Đang tìm kiếm các hồ sơ chứa ít nhất 1 trong các từ khóa: {keywords}',
                additional_data={'text': f'Đang tìm kiếm các hồ sơ chứa ít nhất 1 trong các từ khóa: {keywords}'}
            )
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
        print("\nKhông tìm thấy hồ sơ nào khớp với từ khóa. Sẽ tìm kiếm bằng vector search trên toàn bộ dữ liệu.")
        if user:
            from notifications.utils import create_notification
            create_notification(
                user=user,
                notification_type='profile_creating',
                content=f'Không tìm thấy hồ sơ nào khớp với từ khóa: {keywords}. Sẽ tìm kiếm bằng vector search trên toàn bộ dữ liệu.',
                additional_data={'text': f'Không tìm thấy hồ sơ nào khớp với từ khóa: {keywords}. Sẽ tìm kiếm bằng vector search trên toàn bộ dữ liệu.'}
            )
        keyword_match_counts = {idx: 0 for idx in df_original.index}
    else:
        print(f"\nTìm thấy {len(valid_keyword_indices)} hồ sơ khớp với ít nhất một từ khóa.")

    # --- Bước 2: Thực hiện vector search trên Qdrant ---
    print("\nĐang tạo embedding cho truy vấn...")
    if user:
        from notifications.utils import create_notification
        create_notification(
            user=user,
            notification_type='profile_creating',
            content=f'Đang tạo mã hóa cho truy vấn: {user_query}',
            additional_data={'text': f'Đang tạo mã hóa cho truy vấn: {user_query}'}
        )
    query_embedding = get_embedding(user_query, task_type="RETRIEVAL_QUERY")

    if query_embedding is None:
        print("Lỗi: Không thể tạo embedding cho truy vấn. Sử dụng chỉ kết quả từ khóa.")
        if not valid_keyword_indices:
            return None
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
        verified_indices_str = parallel_verify(user_query, profiles_for_llm, max_profiles=len(profiles_for_llm))
        return verified_indices_str

    # Thực hiện vector search trên Qdrant
    print("Đang thực hiện vector search trên Qdrant...")
    if user:
        from notifications.utils import create_notification
        create_notification(
            user=user,
            notification_type='profile_creating',
            content=f'Đang thực hiện vector search trên Qdrant...',
            additional_data={'text': f'Đang thực hiện vector search trên Qdrant...'}
        )
    
    try:
        # Query Qdrant - không cần lấy collection info (tránh version mismatch)
        # Sử dụng limit lớn để lấy nhiều kết quả nhất có thể
        search_results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=10000  # Giới hạn tối đa 10000 vectors
        )
    except Exception as e:
        print(f"Lỗi khi truy vấn Qdrant: {e}")
        return None

    # --- Bước 3: Kết hợp kết quả từ khóa và vector ---
    print("Đang kết hợp kết quả từ khóa và vector...")
    if user:
        from notifications.utils import create_notification
        create_notification(
            user=user,
            notification_type='profile_creating',
            content=f'Đang kết hợp kết quả từ khóa và vector...',
            additional_data={'text': f'Đang kết hợp kết quả từ khóa và vector...'}
        )

    # Mapping: Qdrant point ID -> Database profile ID
    # Lấy ID thực từ payload (metadata) thay vì dùng point ID
    vector_distances = {}  # Key: DataFrame index, Value: vector score
    db_id_to_df_index = {}  # Map từ database ID sang DataFrame index
    
    # Tạo mapping từ database ID sang DataFrame index một lần
    if 'id' in df_original.columns:
        for df_idx, row in df_original.iterrows():
            db_id = row.get('id')
            if db_id is not None:
                try:
                    db_id = int(db_id)
                    db_id_to_df_index[db_id] = df_idx
                except (ValueError, TypeError):
                    pass
    
    if search_results:
        for result in search_results:
            try:
                qdrant_point_id = int(result.id)
                # Lấy ID thực từ payload (metadata) - đây là ID từ database
                payload = result.payload if hasattr(result, 'payload') else {}
                db_id_str = payload.get('id', '')
                
                if db_id_str:
                    try:
                        db_id = int(db_id_str)
                        # Tìm DataFrame index từ database ID
                        if db_id in db_id_to_df_index:
                            df_idx = db_id_to_df_index[db_id]
                            vector_distances[df_idx] = result.score
                        else:
                            # Fallback: thử tìm trực tiếp trong DataFrame
                            matching_rows = df_original[df_original['id'] == db_id]
                            if not matching_rows.empty:
                                df_idx = matching_rows.index[0]
                                vector_distances[df_idx] = result.score
                                db_id_to_df_index[db_id] = df_idx
                    except (ValueError, TypeError):
                        # Nếu không parse được, bỏ qua
                        continue
                else:
                    # Nếu không có ID trong payload, thử dùng point ID (fallback)
                    # Chỉ dùng nếu point ID trùng với database ID
                    if qdrant_point_id in db_id_to_df_index:
                        df_idx = db_id_to_df_index[qdrant_point_id]
                        vector_distances[df_idx] = result.score
            except (ValueError, AttributeError) as e:
                print(f"Lỗi khi xử lý kết quả từ Qdrant: {e}")
                continue
    else:
        print("Không nhận được kết quả từ vector search.")
        return None

    combined_scores = {}
    KEYWORD_BONUS = 0.05

    for idx in df_original.index:
        vector_score = vector_distances.get(idx, 0)
        keyword_count = keyword_match_counts.get(idx, 0)
        total_score = vector_score + keyword_count * KEYWORD_BONUS
        if total_score > 0:
            combined_scores[idx] = total_score

    if not combined_scores:
        print("Không tìm thấy hồ sơ nào có điểm kết hợp > 0.")
        return None

    print(f"Đang xếp hạng {len(combined_scores)} hồ sơ theo điểm kết hợp...")
    ranked_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    top_results = ranked_results[:top_n_final]

    print(f"\n--- Top {min(10, len(top_results))} Kết quả (Theo Điểm Kết Hợp, Trước LLM) ---")
    if user:
        from notifications.utils import create_notification
        create_notification(
            user=user,
            notification_type='profile_creating',
            content=f'Đã tìm thấy {len(top_results)} hồ sơ phù hợp sau khi kết hợp từ khóa và vector search.',
            additional_data={'text': f'Đã tìm thấy {len(top_results)} hồ sơ phù hợp sau khi kết hợp từ khóa và vector search.'}
        )

    profiles_for_llm = []
    for idx, _ in top_results:
        try:
            profile_data = df_original.loc[idx].copy()
            profile_data['id'] = str(idx)
            profiles_for_llm.append(profile_data)
        except KeyError:
            continue

    if not profiles_for_llm:
        print("Không tìm thấy hồ sơ hợp lệ để gửi đến LLM.")
        return None

    print(f"\nĐang xác minh {len(profiles_for_llm)} kết quả với Gemini LLM...")
    if user:
        from notifications.utils import create_notification
        create_notification(
            user=user,
            notification_type='profile_creating',
            content=f'Đang xác minh {len(profiles_for_llm)} kết quả với Gemini LLM...',
            additional_data={'text': f'Đang xác minh {len(profiles_for_llm)} kết quả với Gemini LLM...'}
        )
    verified_indices_str = parallel_verify(user_query, profiles_for_llm, max_profiles=len(profiles_for_llm))

    if verified_indices_str:
        print(f"\n=== {len(verified_indices_str)} KẾT QUẢ PHÙ HỢP NHẤT SAU KHI LỌC BẰNG LLM ===")
        if user:
            from notifications.utils import create_notification
            create_notification(
                user=user,
                notification_type='profile_creating',
                content=f'Đã tìm thấy {len(verified_indices_str) - 1} kết quả phù hợp nhất sau khi lọc bằng LLM.',
                additional_data={'text': f'Đã tìm thấy {len(verified_indices_str) - 1} kết quả phù hợp nhất sau khi lọc bằng LLM.'}
            )
        verified_indices_int = [int(id_str) for id_str in verified_indices_str if id_str.isdigit()]
        verified_with_scores = [(idx, combined_scores.get(idx, 0)) for idx in verified_indices_int if idx in combined_scores]
        verified_with_scores.sort(key=lambda x: x[1], reverse=True)

        for i, (idx, score) in enumerate(verified_with_scores):
            try:
                profile = df_original.loc[idx]
                kw_count = keyword_match_counts.get(idx, 0)
                kw_score = kw_count * KEYWORD_BONUS
                vec_score = vector_distances.get(idx, 0)
                print(f"\n{i+1}. ID: {idx}")
                print(f"Tổng điểm: {score:.4f} (Vector: {vec_score:.4f}, KW: {kw_score:.4f}, Từ khóa khớp: {kw_count})")
                print(f"Tiêu đề: {profile.get('Tiêu đề', 'N/A')}")
                print(f"Họ và tên: {profile.get('Họ và tên', 'N/A')}")
                print(f"Năm thất lạc: {profile.get('Năm thất lạc', 'N/A')}")
                print(f"Năm sinh: {profile.get('Năm sinh', 'N/A')}")
                print(f"Chi tiết: {str(profile.get(DETAIL_COLUMN_NAME, 'N/A'))[:500]}...")
                print("-" * 60)
            except Exception as e:
                print(f"Lỗi khi hiển thị hồ sơ {idx}: {e}")
    else:
        print("\nKhông tìm thấy hồ sơ nào phù hợp sau khi xác minh bằng LLM.")

    if return_json:
        result_list = []
        for idx in verified_indices_str:
            try:
                idx_int = int(idx)
                profile = df_original.loc[idx_int]
                # Lấy ID thực từ database (từ cột 'id' hoặc dùng index nếu index là ID)
                if 'id' in df_original.columns:
                    db_id = profile.get('id', idx_int)
                    # Đảm bảo db_id là số nguyên
                    try:
                        db_id = int(db_id) if db_id is not None else idx_int
                    except (ValueError, TypeError):
                        db_id = idx_int
                else:
                    db_id = idx_int
                
                result_list.append({
                    "id": str(db_id),  # Dùng ID thực từ database
                    "total_score": combined_scores.get(idx_int, 0),
                    "vector_score": vector_distances.get(idx_int, 0),
                    "keyword_score": keyword_match_counts.get(idx_int, 0) * KEYWORD_BONUS,
                    "matched_keywords": keyword_match_counts.get(idx_int, 0),
                    "title": profile.get('Tiêu đề', ''),
                    "full_name": profile.get('Họ và tên', ''),
                    "losing_year": profile.get('Năm thất lạc', ''),
                    "born_year": profile.get('Năm sinh', ''),
                    "name_of_father": profile.get('Tên cha', ''),
                    "name_of_mother": profile.get('Tên mẹ', ''),
                    "siblings": profile.get('Anh chị em', ''),
                    "detail": str(profile.get(DETAIL_COLUMN_NAME, '')),
                    "link": profile.get('Link', ''),
                })
            except Exception as e:
                print(f"Lỗi khi build kết quả JSON: {e}")
                continue
        return result_list

    return verified_indices_str