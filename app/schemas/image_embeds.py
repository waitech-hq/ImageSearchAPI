from pydantic import BaseModel
 


class ImageEmbed(BaseModel):
    id: int
    # annoy_index: int
    image_path: str
    embedding: str
   