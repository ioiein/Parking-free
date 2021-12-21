import os
from fastapi import FastAPI
from fastapi.responses import FileResponse
import uvicorn
import json


app = FastAPI()


@app.get("/")
async def main():
    return FileResponse('img/map.jpg')


@app.get("/origin")
async def original_img():
    return FileResponse('img/out.jpg')


@app.get("/slots")
async def free_slots():
    with open('img/slots.json', 'r') as f:
        slots = json.load(f)
    return slots

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=os.getenv("PORT", 8000))
