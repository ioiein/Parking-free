import os
from fastapi import FastAPI
from fastapi.responses import FileResponse
import uvicorn


app = FastAPI()


@app.get("/")
async def main():
    return FileResponse('img/smth.jpg')

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=os.getenv("PORT", 8000))