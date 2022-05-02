import sys
sys.dont_write_bytecode = True
# pycache remove
import uvicorn
from fastapi import FastAPI
from .api.v1.api import api_router
from .core.config import settings
# from .models import user, reservation, restaurant
from .db import database


# user.Base.metadata.create_all(bind=database.engine)

app = FastAPI(title=settings.PROJECT_NAME)
app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get('/api/v1/')
def root():
    return {"message": "Hello, You've reached the root of your project"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8600)