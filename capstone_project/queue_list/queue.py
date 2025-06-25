import threading

# Hàng đợi người dùng và lock
user_queue = []
queue_lock = threading.Lock()

def add_user_to_queue(user_id):
    """Thêm người dùng vào hàng đợi."""
    with queue_lock:
        user_queue.append(user_id)
        print(f"User {user_id} added to queue. Current queue: {user_queue}")

def remove_user_from_queue(user_id):
    """Xóa người dùng khỏi hàng đợi sau khi tìm kiếm xong."""
    with queue_lock:
        if user_id in user_queue:
            user_queue.remove(user_id)
            print(f"User {user_id} removed from queue. Current queue: {user_queue}")
        else:
            print(f"User {user_id} not found in queue.")

def get_next_user_from_queue():
    """Lấy người dùng tiếp theo từ hàng đợi."""
    with queue_lock:
        if user_queue:
            return user_queue[0]  # Lấy người dùng đầu tiên
        else:
            return None

def is_user_turn(user_id):
    """Kiểm tra xem có phải lượt của người dùng này không."""
    with queue_lock:
        if user_queue and user_queue[0] == user_id:
            return True
        else:
            return False