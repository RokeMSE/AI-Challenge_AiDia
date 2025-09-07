import os
import csv
import re
import pandas as pd

MAP_FRAME_B1 = "/Users/dangnguyen/Desktop/AI-Challenge_AiDia/data/map_keyframes/map_keyframes_b1"
MAP_FRAME_B2 = "/Users/dangnguyen/Desktop/AI-Challenge_AiDia/data/map_keyframes/map_keyframes_b2"

part = "p2"

def extract_query_info(query_filename):
    """
    Extract query ID and type from the query filename.
    Example: query-p2-15-qa.txt -> (15, 'qa')
    """
    pattern = r'query-p2-(\d+)-([a-z]+)\.txt'
    match = re.match(pattern, query_filename)
    if match:
        query_id = int(match.group(1))
        query_type = match.group(2)
        return query_id, query_type
    return None, None

def ensure_result_folder():
    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    result_dir = os.path.join(workspace_root, 'result')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    return result_dir


# Write KIS results to CSV
def write_kis_results(results, filename):
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for r in results:
            writer.writerow([r.video_id, r.frame_index])

# Write Q&A results to CSV
def write_qa_results(results, filename):
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for r in results:
            writer.writerow([r.video_id, r.frame_index, r.answer])

# Write TRAKE results to CSV
def write_trake_results(results, filename):
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for r in results:
            row = [r.video_id] + r.frame_ids
            writer.writerow(row)

def save_kis_results(results, query_id):
    result_dir = ensure_result_folder()
    filename = os.path.join(result_dir, f"query-{part}-{query_id}-kis.csv")
    write_kis_results(results, filename)

def save_qa_results(results, query_id):
    result_dir = ensure_result_folder()
    filename = os.path.join(result_dir, f"query-{part}-{query_id}-qa.csv")
    write_qa_results(results, filename)

def save_trake_results(results, query_id):
    result_dir = ensure_result_folder()
    filename = os.path.join(result_dir, f"query-{part}-{query_id}-trake.csv")
    write_trake_results(results, filename)

def create_kis_result_object(video_id, frame_index):
    return type('KISResult', (object,), {
        'video_id': video_id, 
        'frame_index': frame_index
    })()

def create_qa_result_object(video_id, frame_index, answer):
    return type('QAResult', (object,), {
        'video_id': video_id, 
        'frame_index': frame_index, 
        'answer': answer
    })()

def create_trake_result_object(video_id, frame_ids):
    return type('TrakeResult', (object,), {
        'video_id': video_id, 
        'frame_ids': frame_ids
    })()

def map_keyframes(video_id, n_value):
    if video_id[0] == 'L':
        batch = 1
    elif video_id[0] == 'K':
        batch = 2
    else:
        raise ValueError("Waiting for update...")

    file_path = os.path.join(MAP_FRAME_B1 if batch == 1 else MAP_FRAME_B2, f"{video_id}.csv")

    df = pd.read_csv(file_path)
    if 0 <= int(n_value) <= len(df):
        return int(df.iloc[int(n_value)-1]['frame_idx'])
    else:
        return None