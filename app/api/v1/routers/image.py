from typing import List
from app.models.user import User as UserModel
from app.schemas import user as user_schema
from fastapi import  status, HTTPException, Depends, APIRouter
from sqlalchemy.orm import Session
from app.api.deps import get_db 


router = APIRouter()

# @router.get('/', response_model=List[user_schema.UserOut])
# def get_texts(db: Session = Depends(get_db)):

#      users = db.query(UserModel).all()

#      if not users:
#           raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f'User with id: {id} was not found')
#      return users 


@router.get('/', response_model=user_schema.UserOut)
def search_text(db: Session = Depends(get_db)):
     return {'msg': 'Searching with text'}