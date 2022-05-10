import sys, os, ssl, glob

# pycache remove
import uvicorn
from fastapi import FastAPI
from .api.v1.api import api_router
from .core.config import settings
from .models import image_embeds
from .db import database
from fastapi.staticfiles import StaticFiles

sys.dont_write_bytecode = True
ssl._create_default_https_context = ssl._create_unverified_context


image_embeds.Base.metadata.create_all(bind=database.engine)
script_dir = os.path.dirname(__file__)
st_abs_file_path = os.path.join(script_dir, "static/")

# FastAPI
app = FastAPI(title=settings.PROJECT_NAME)
app.include_router(api_router, prefix=settings.API_V1_STR)
app.mount("/static", StaticFiles(directory=st_abs_file_path), name="static")


@app.get('/api/v1/')
def root():
    
    return {"message": "Hello, You've reached the root of your project"}



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8600)