"""
Helper để khởi tạo Pinecone Index (không phụ thuộc Django settings).
Yêu cầu env:
  - PINECONE_API_KEY
  - PINECONE_INDEX_HOST (ưu tiên, dạng https://XXXX-XXXX.svc.XXXX.pinecone.io)
  - hoặc PINECONE_INDEX_NAME (nếu dùng cách cũ, cần thêm project/env; ở đây ưu tiên host)
"""

import os
from typing import Optional

from pinecone import Pinecone

from .config import PINECONE_API_KEY, PINECONE_INDEX_HOST, PINECONE_INDEX_NAME


def get_pinecone_index():
    """
    Trả về pinecone.Index đã kết nối sẵn.
    Ưu tiên dùng host (SDK v3). Nếu thiếu API key hoặc host thì trả None.
    """
    if not PINECONE_API_KEY:
        print("Pinecone: thiếu PINECONE_API_KEY")
        return None

    pc = Pinecone(api_key=PINECONE_API_KEY)

    if PINECONE_INDEX_HOST:
        try:
            return pc.Index(host=PINECONE_INDEX_HOST)
        except Exception as e:
            print(f"Pinecone: lỗi khi kết nối host {PINECONE_INDEX_HOST}: {e}")
            return None

    if PINECONE_INDEX_NAME:
        try:
            return pc.Index(PINECONE_INDEX_NAME)
        except Exception as e:
            print(f"Pinecone: lỗi khi mở index {PINECONE_INDEX_NAME}: {e}")
            return None

    print("Pinecone: chưa khai báo PINECONE_INDEX_HOST hoặc PINECONE_INDEX_NAME")
    return None



