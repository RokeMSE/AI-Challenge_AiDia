# AI CHALLENGE: AiĐia
- Framework dựa trên model CLIP4clip

## Dir format for data:
- Để sử dụng tool, set format của data như sau (hoặc là reformat lại code trong để đọc/scan file theo kiểu khác)
```
data/
  ├── video_frames/           # Original video frames (Download from the BTC Excel file)`
  │   ├── L01_V001/
  │   │   ├── 001.jpg
  │   │   ├── 002.jpg
  │   │   └── ...
  │   └── L01_V002/
  ├── embeddings/            # CLIP embeddings (generated)
  │   ├── embeddings_L01_V001/
  │   │   ├── 001.npy
  │   │   ├── 002.npy
  │   │   └── ...
  │   └── embeddings_L01_V002/
  └── object/     # Object detection JSON files (Download from the BTC Excel file)
      ├── L01_V001/
      │   ├── 001.json        # == 001.jpg
      │   ├── 002.json        # == 002.jpg
      │   └── ...
      └── L01_V002/
```

# About `modules/db`
- Sử dụng weaviate vector database được deploy trên docker
- Trong file `db.py` có cung cấp một class hỗ trợ kết nối với weaviate và thực hiện thao tác: tạo collection, insert, search.
- Để chạy:
  - (Đảm bảo đã tải Docker về máy) Trong directory chứa file `docker-compose.yml`, chạy lệnh `docker compose up --build -d` để tạo container
  - Trong client code, để sử dụng, trước tiên cần khởi tạo một instance của class `WeaviateRepository`. Khi khởi tạo, mặc định class sẽ tạo collection mới nếu chưa tồn tại; nếu đã tồn tại rồi thì tiếp tục làm việc trên cái cũ.
  - Thao tác đọc vector embedding để lưu lên db:
    - Để upload toàn bộ vector trong `data/embeddings`: chỉ cần cung cấp đường dẫn ***absolute path*** đến directory `data/embeddings`, chuyện còn lại là chỉ cần gọi method `upload_from_folder()`
    - Để upload ***tất cả frame*** của *chỉ 1* clip: gọi hàm `upload_from_npy(...)` và tham khảo cách sử dụng hàm trong code của `upload_from_folder()`
  - Thao tác search (***k-NN***):
    - Nếu search theo vector: cần cung cấp sẵn một vector và một giá trị k. Kết quả trả về sẽ là một list các `QueryResult` (tham khảo dataclass `QueryResult`)
    - Nếu search theo text: cung cấp câu query theo text và giá trị k. Kết quả trả về sẽ là một list các `QueryResult` (tham khảo dataclass `QueryResult`)
- Đây là đoạn sample code hướng dẫn sử dụng:
```python
try:
  repos = WeaviateRepository()
except Exception as e:
  print(f"Failed to connect to Weaviate at. Is it running?")
  print(e)
  exit(1)

# test    
DATA_ROOT = "/home/ketamean/Documents/Y3/AIC/AI-Challenge_AiDia/data/embeddings"
repos.upload_from_folder(DATA_ROOT, batch_size=50)

res = repos.query_by_vector(np.load(os.path.join(DATA_ROOT, "L21_V002", '023.npy'))[0].tolist(), k=5)
print(res)
```

## Delete Huggingface Models:
`pip install huggingface_hub["cli"]` -> `huggingface-cli delete-cache`