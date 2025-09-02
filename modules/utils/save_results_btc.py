import os
import csv
import re

part = "p1"

def extract_query_info(query_filename):
    """
    Extract query ID and type from the query filename.
    Example: query-p1-15-qa.txt -> (15, 'qa')
    """
    pattern = r'query-p1-(\d+)-([a-z]+)\.txt'
    match = re.match(pattern, query_filename)
    if match:
        query_id = int(match.group(1))
        query_type = match.group(2)
        return query_id, query_type
    return None, None

def ensure_result_folder():
    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    result_dir = os.path.join(workspace_root, 'result_btc')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        print(f"Created result directory: {result_dir}")
    return result_dir

def format_frame_identifier(frame_name, vector_index):
    """
    Creates a consistent frame identifier that includes vector index.
    For BTC data: frame_name + "_" + vector_index
    """
    return f"{frame_name}_{vector_index}"

# Write KIS results to CSV
def write_kis_results(results, filename):
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "frame_id"])
        for r in results:
            if '_' in r.frame_index:
                video_id, frame_id = r.frame_index.rsplit('_', 1)
            else:
                video_id, frame_id = r.frame_index, ''
            if int(frame_id) < 100:
                frame_id = f"0{frame_id}"
            writer.writerow([video_id, frame_id])

# Write Q&A results to CSV
def write_qa_results(results, filename):
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "frame_id", "answer"])
        for r in results:
            writer.writerow([r.video_id, r.frame_index, r.answer])

# Write TRAKE results to CSV
def write_trake_results(results, filename):
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Header: video_id, frame_id_1, frame_id_2, ...
        max_frames = max(len(r.frame_ids) for r in results) if results else 0
        header = ["video_id"] + [f"frame_id_{i+1}" for i in range(max_frames)]
        writer.writerow(header)
        for r in results:
            row = [r.video_id] + r.frame_ids
            row += ["" for _ in range(max_frames - len(r.frame_ids))]
            writer.writerow(row)

def save_kis_results(results, query_id):
    result_dir = ensure_result_folder()
    filename = os.path.join(result_dir, f"query-{part}-{query_id}-kis.csv")
    write_kis_results(results, filename)
    print(f"KIS results saved to: {filename}")

def save_qa_results(results, query_id):
    result_dir = ensure_result_folder()
    filename = os.path.join(result_dir, f"query-{part}-{query_id}-qa.csv")
    write_qa_results(results, filename)
    print(f"QA results saved to: {filename}")

def save_trake_results(results, query_id):
    result_dir = ensure_result_folder()
    filename = os.path.join(result_dir, f"query-{part}-{query_id}-trake.csv")
    write_trake_results(results, filename)
    print(f"TRAKE results saved to: {filename}")

def create_kis_result_object(video_id, frame_index):
    """
    Creates KIS result object.
    For BTC: frame_index should include vector index (e.g., "frame001_5")
    """
    return type('KISResult', (object,), {
        'video_id': video_id, 
        'frame_index': frame_index
    })()

def create_qa_result_object(video_id, frame_index, answer):
    """
    Creates QA result object.
    For BTC: frame_index should include vector index (e.g., "frame001_5")
    """
    return type('QAResult', (object,), {
        'video_id': video_id, 
        'frame_index': frame_index, 
        'answer': answer
    })()

def create_trake_result_object(video_id, frame_ids):
    """
    Creates TRAKE result object.
    For BTC: frame_ids should include vector indices (e.g., ["frame001_5", "frame002_3"])
    """
    return type('TrakeResult', (object,), {
        'video_id': video_id, 
        'frame_ids': frame_ids
    })()

def create_btc_kis_result_object(video_id, frame_name, vector_index):
    """
    Creates KIS result object specifically for BTC data structure.
    Combines frame_name and vector_index into a single identifier.
    """
    frame_identifier = format_frame_identifier(frame_name, vector_index)
    return create_kis_result_object(video_id, frame_identifier)

def create_btc_qa_result_object(video_id, frame_name, vector_index, answer):
    """
    Creates QA result object specifically for BTC data structure.
    Combines frame_name and vector_index into a single identifier.
    """
    frame_identifier = format_frame_identifier(frame_name, vector_index)
    return create_qa_result_object(video_id, frame_identifier, answer)

def create_btc_trake_result_object(video_id, frame_data_list):
    """
    Creates TRAKE result object specifically for BTC data structure.
    frame_data_list: list of tuples (frame_name, vector_index)
    """
    frame_identifiers = [format_frame_identifier(frame_name, vector_index) 
                        for frame_name, vector_index in frame_data_list]
    return create_trake_result_object(video_id, frame_identifiers)