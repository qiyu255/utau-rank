import csv
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


def make_check(tid, typ):
    with open(OCR_OUTPUT/f'{tid}_{typ}.csv', encoding='utf-8') as f:
        info = list(csv.DictReader(
            f, fieldnames=FIELDNAMES[typ], delimiter=';'))

    table = []
    for i, row in enumerate(info):
        for label, value in row.items():
            if label == None:
                continue
            item = {
                'nu': f'{i:0>3}',
                'label': label,
                'value': label == 'title' and short_title(value) or value,
            }
            if typ == 'info':
                ite_ = {
                    'nu': f'{i:0>3}',
                    'label': label[:-2]+'__',
                    'value': value,
                }
                table.append(ite_)

            # TODO: preprocess title
            table.append(item)

    return table


def short_title(t):
    t = re.sub(r'(\[.*?\]|【.*?】)', '', t)
    return t


def sort_key(x):
    return x['label'][:-2]


def main():
    info = make_check(tid, 'info')
    data = make_check(tid, 'data')
    with open(CHECK_DIR/f'{tid}.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['nu', 'label', 'value'])
        w.writeheader()
        w.writerows([*sorted(info, key=sort_key), *sorted(data, key=sort_key)])


if __name__ == '__main__':
    main()
