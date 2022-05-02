import email
from ..db.database import Base
from sqlalchemy import Column, Integer, String, Boolean
from sqlalchemy.sql.sqltypes import TIMESTAMP
from sqlalchemy.sql.expression import text


class ImageEmbeds(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, nullable=False)
    annoy_index = Column(Integer, nullable=False, unique=True) #future
    image_path = Column(String, nullable=False)
    embedding = Column(String, nullable=False)

