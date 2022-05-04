from email.mime import application
from typing import List
from app.models.image_embeds import ImageEmbed as ImageEmbedModel
from app.schemas import image_embeds as embed_schema
from fastapi import  Request, status, HTTPException, Depends, APIRouter
from sqlalchemy.orm import Session
from app.api.deps import get_db 
from app.db.database import engine

import glob, torch, clip, shutil
import numpy as np
import pandas as pd

from fastapi.responses import HTMLResponse


from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, preprocess = clip.load("ViT-B/32", device=device)

CONTENT_STORE = '/Users/elvischege/Desktop/Python/FastAPI/ImageSearchEngine/./app/static/'
print("Loading Model...")


router = APIRouter()

 

# Image & Text Functions
def preprocess_image(image_path):
	image = Image.open(image_path)
	image = preprocess(image).unsqueeze(0).to(device)
	return image


def preprocess_text(text):
	return clip.tokenize(text).to(device)


@router.get('/')
def search_text(db: Session = Depends(get_db)):
     return {'msg': 'Searching with text'}


def create_image_embeddings(db, image_paths):
	
	embed_template = {
          'image_path': str,
          'embedding': str
          }

	for img_path in image_paths:
          
		processed_image = preprocess_image(img_path)
		with torch.no_grad():
			embed = model.encode_image(processed_image).detach().numpy()
		embed = embed.tostring()

		embed_template['embedding'] = embed
		embed_template['image_path'] = img_path

		new_embed = ImageEmbedModel(**embed_template)
		embed_template.clear()

		db.add(new_embed)
		db.commit()
		db.refresh(new_embed)


	return True


def cal_sim(feat1, feat2):


	img_embed = np.fromstring(feat2, dtype=np.float32)
	img_embed = img_embed.reshape((1, img_embed.shape[0]))
	sim = cosine_similarity(feat1, img_embed)
	return sim[0][0]

def text_images_similarity(text, df):
	## Preprocess text
	processed_text = preprocess_text([text])
	
	with torch.no_grad():
		text_embed = model.encode_text(processed_text)
	

	
	# print(f"Heyy and ummm {df['embedding']}")
	## Calculate cos sim for all images wrt to text
	# df['sim'] = df['embedding'].apply(lambda x: cal_sim(text_embed, x))
	application_x = lambda x: cal_sim(text_embed, x) # THIS IS WHERE THE BUGG IS AHHHH

	print('THE SIMM WORKSS ahhhh', df['embedding'].apply(application_x))
	
	
	# df = df.sort_values(by=['sim'], ascending=False)
	# return df
	return 0

def image_images_similarity(img_path, df):
	## Preprocess image
	processed_image = preprocess_image(img_path)
	with torch.no_grad():
		image_embed = model.encode_image(processed_image).detach().numpy()

	df['sim'] = df['embedding'].apply(lambda x: cal_sim(image_embed, x))

	df = df.sort_values(by=['sim'], ascending=False)
	return df
	
def get_image_data(db):
     
     image_embeds = db.query(ImageEmbedModel).all()
     if not image_embeds:
          raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
	
     return image_embeds

# Pandas Read SQL with SQLALCHEMY ORM CONVERSION
def get_image_data_df():

     df = pd.read_sql_table('image_embeds', con=engine)
     return df

# NEEDS SERIOUS TESTING

@router.get("/create_image_embeds")
def create_image_embeds(db: Session = Depends(get_db)):
	
	image_paths = glob.glob(CONTENT_STORE+'/*.jpeg')
	print('PATHS:', image_paths)
	status = create_image_embeddings(db, image_paths)
	if status:
		return "Inserted"
	else:
		return "Failed"


# Acts as image base search
@router.get("/imagesearchhome")
def imagesearchhome():
	
	df = get_image_data_df()
	# images = df['image_path'].tolist()
	
	return {"images_in_db": df['image_path'].to_dict()}

@router.post("/imagesearch")
async def imagesearch(search_input: str):

	
	print(f'\n\n\n{search_input}\n\n')

	## Get whole data from DB
	df = get_image_data_df()
	df_sim = text_images_similarity(search_input, df) # issue is here
	print('df smmm', df_sim)
	# images = df_sim['image_path'].tolist()

	# context = {"request": request, "images": images, "text": text}
	# print("\n\n\n DIS IS DA TEXT", text, '\n\n')
	return "Hello"