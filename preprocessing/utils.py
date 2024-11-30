import json
import re

def split_work_experience(text):
    works = re.split(r"(?=\d{4}-\d{2}-\d{2}\s-\s(?:\d{4}-\d{2}-\d{2}|:))", text)
    return set(entry.strip() for entry in works if entry.strip())