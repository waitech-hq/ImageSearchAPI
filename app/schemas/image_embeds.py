from pydantic import BaseModel
from typing import Any 


class ImageEmbed(BaseModel):
    id: int
    # annoy_index: int
    image_path: str
    embedding: Any # I dunno what datatype binary is sooooo
   