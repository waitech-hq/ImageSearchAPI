from pydantic import BaseModel
from torch import embedding


class ImageEmbed(BaseModel):
    id: int
    image_path: str
    embedding: str
   