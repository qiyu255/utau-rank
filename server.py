from fastapi import FastAPI, HTTPException, Body, Path
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
import os
import json
from pathlib import Path as PathlibPath

app = FastAPI()

app.mount("/video", StaticFiles(directory="video"), name="video")
app.mount("/cache", StaticFiles(directory="cache"), name="cache")
app.mount("/pages", StaticFiles(directory="pages"), name="pages")

DATA_DIR = PathlibPath("data")
SAMPLE_CACHE_DIR = PathlibPath("cache/sample")

def ensure_data_dir():
    DATA_DIR.mkdir(exist_ok=True)

@app.get("/sample")
async def get_sample_list():
    sample_dict = {}
    if SAMPLE_CACHE_DIR.exists() and SAMPLE_CACHE_DIR.is_dir():
        for file_path in SAMPLE_CACHE_DIR.glob("*.png"):
            name = file_path.stem
            sample_dict[name] = f"/cache/sample/{name}.png"
    return sample_dict

@app.get("/roi/{name}")
async def get_roi(name: str = Path(..., description="ROI 名称")):
    ensure_data_dir()
    file_path = DATA_DIR / f"roi_{name}.json"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"ROI file for '{name}' not found")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Invalid JSON in file")

@app.put("/roi/{name}")
async def update_or_create_roi(
    name: str = Path(..., description="ROI 名称"),
    roi_data: dict = Body(..., description="ROI 数据（JSON 对象）")
):
    ensure_data_dir()
    file_path = DATA_DIR / f"roi_{name}.json"
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(roi_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write file: {str(e)}")
    return {"message": f"ROI data for '{name}' saved successfully", "file": str(file_path)}

@app.get("/")
def root():
    return RedirectResponse(url="/pages/index.html")