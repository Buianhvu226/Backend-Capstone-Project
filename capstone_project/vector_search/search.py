import numpy as np
import pandas as pd
import time
import math

from .config import DETAIL_COLUMN_NAME
from .embedding import get_embedding
from .llm_utils import extract_keywords_gemini, parallel_verify

def search_combined_chroma(df_original, collection, user_query, top_n_final=200, return_json=False, user=None):
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
                content=f'{len(verified_indices_str)} kết quả phù hợp nhất sau khi lọc bằng LLM.',
                additional_data={
                    'text': f'{len(verified_indices_str)} kết quả phù hợp nhất sau khi lọc bằng LLM.',
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
            content=f'Đã tìm thấy {len(verified_indices_str)} kết quả phù hợp nhất sau khi lọc bằng LLM.',
            additional_data={
                'text': f'Đã tìm thấy {len(verified_indices_str)} kết quả phù hợp nhất sau khi lọc bằng LLM.',
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