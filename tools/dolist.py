import json
import sys
sys.stdout.reconfigure(encoding='utf-8')
def info(payload):
    id = payload["id"]
    title = payload["title"]
    if 'VOCAL' in title:
        return
    print(f'{title} = {id}')

def dlcmd(payload):
    id = payload["id"]
    title = payload["title"]
    if 'VOCAL' in title:
        return
    print(f'yt-dlp --cookies cookies.txt -f bestvideo+bestaudio --merge-output-format mp4 https://www.bilibili.com/video/av{id} -o video/av{id}.mp4')


payloads = json.load(open("data/ml426206421.json", "r", encoding="utf-8"))

for payload in payloads:
    info(payload)
    # dlcmd(payload)