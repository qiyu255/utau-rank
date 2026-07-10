import csv
import json
import re
from pathlib import Path

tid = 103
INFO_LABELS = ["title", "library", "author"]
DATA_LABELS = ["rank", "last", "total", "view", "comment",
               "mylist", "mylist_rate", "comment_rate", "date", "id"]
FIELDNAMES = {'info': INFO_LABELS, 'data': DATA_LABELS}
OCR_INPUT = Path('cache/layout')
OCR_OUTPUT = Path('cache/OCR-OUTPUT')
CHECK_DIR = Path('cache/check')
CHECK_DIR.mkdir(parents=True, exist_ok=True)


def load_check(tid):
    with open(CHECK_DIR/f'{tid}.csv', encoding='utf-8') as f:
        it = csv.DictReader(f) # fieldnames=['nu', 'label', 'value']

        rows = []
        for row in it:
            if row['label'].endswith('_'):
                continue
            print(row['nu'])
            row['nu'] = int(row["nu"])
            rows.append(row)

    return rows


def setup_theme(rows, tid):
    with open(OCR_INPUT/f'info-theme-track.json', encoding='utf-8') as f:
        theme_track = json.load(f)
    for i in rows:
        i['theme'] = theme_track[i['nu']][0]


def group(data):
    temp = {}
    for item in data:
        t = item['theme']
        id_ = item['nu']
        key = item['label']
        value = item['value']
        if t not in temp:
            temp[t] = {}
        if id_ not in temp[t]:
            temp[t][id_] = {'nu': id_}  
        temp[t][id_][key] = value

    result = {}
    for t, id_dict in temp.items():
        result[t] = list(id_dict.values())

    return result

def main():
    data = load_check(tid)
    setup_theme(data, tid)
    data = group(sorted(data,key=lambda x:x['theme']))

    with open(OCR_OUTPUT/f'{tid}_ready.json','w', encoding='utf-8') as f:
        json.dump(data,f, ensure_ascii=False, indent=2)

    print(data)



if __name__ == '__main__':
    main()

