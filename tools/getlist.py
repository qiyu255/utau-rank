import requests
import json

headers = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/138.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.bilibili.com/",
    "Origin": "https://www.bilibili.com/",
    "Accept": "application/json, text/plain, */*",
}

url = "https://api.bilibili.com/x/v3/fav/resource/list"
media_id = "426206421"


pn = 1
ps = 20

payloads = []

while True:
    r = requests.get(
        "https://api.bilibili.com/x/v3/fav/resource/list",
        params={
            "media_id": media_id,
            "pn": pn,
            "ps": ps,
            "platform": "web",
        },
        headers=headers,
    )
    print(pn, ps, r.status_code)

    j = r.json()

    if j["code"] != 0:
        raise RuntimeError(j)

    medias = j["data"]["medias"]

    if not medias:
        break

    payloads.extend(medias)

    print(f"page={pn}, total={len(payloads)}")

    pn += 1

print("全部数量：", len(payloads))

with open(f"data/ml{media_id}.json", "w", encoding="utf-8") as f:
    f.write(json.dumps(payloads, ensure_ascii=False, indent=4))