import chardet
import pandas as pd
from io import StringIO
import os

def read_csv_with_encoding(filepath):
    with open(filepath, 'rb') as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding'] if result['encoding'] else 'utf-8'

    try:
        with open(filepath, 'r', encoding=encoding, errors='replace') as f:
            file_content = f.read()

        df = pd.read_csv(StringIO(file_content), low_memory=False)
    except Exception as e:
        return f"Error reading the file: {e}"

    return df

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv'}
