from fastapi import APIRouter, Depends, status, Response, HTTPException
from sqlalchemy.orm import Session
from fastapi.security.oauth2 import OAuth2PasswordRequestForm
from app.api.deps import get_db
from app.models.user import User as UserModel

router = APIRouter()

@router.get('/text')
def search_image(db: Session = Depends(get_db)):
    return {'msg': 'Searching image'}

    
