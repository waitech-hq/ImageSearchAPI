from pydantic import BaseModel
 


class ImageEmbed(BaseModel):
    id: int
    image_path: str
    embedding: str
   