from fastapi import APIRouter, Depends, status, Response, HTTPException
from sqlalchemy.orm import Session
from fastapi.security.oauth2 import OAuth2PasswordRequestForm
from app.api.deps import get_db
from app.models.image_embeds import User as UserModel

router = APIRouter()

@router.get('/')
def search_image(db: Session = Depends(get_db)):
    return {'msg': 'Searching image'}

    
