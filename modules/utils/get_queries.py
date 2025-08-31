from dataclasses import dataclass
from sentence_transformers import SentenceTransformer

@dataclass
class Query:
  query_name: str
  query_text: str

class QueryLoader:
  def __init__(self, query_root_abs_path: str):
    pass

  def retrieve(self) -> list[Query]:
    # read the folder containing 25 .txt queries from the organizer and extract json
    ...
    # temporary code returns 3 queries
    return [
      Query(
        query_text="Đoạn video về một chương trình từ thiện của một câu lạc bộ tên là FANA. Trong đoạn video có thể thấy câu lạc bộ này đang đi trao quà tại một xã thuộc tỉnh Khánh Hòa. Hỏi xã này có tên là gì? (tại thời điểm đó)",
        query_name="Query 1"
      ),
      Query(
        query_text="Đây là phần giới thiệu việc phóng tàu vũ trụ tư nhân. Đoạn clip bắt đầu với hình ảnh 4 phi hành gia mặc áo đen. Một trong những nhiệm vụ dự kiến của tàu vũ trụ là nghiên cứu ánh sáng cực quang ở vùng cực",
        query_name="Query 2"
      ),
      Query(
        query_text="Mẩu tin giới thiệu về đàn hổ tại một địa phương ở miền Nam vừa có thêm khoảng 3-6 con hổ con. Đây là một giống hổ quý hiếm",
        query_name="Query 3"
      )
    ]