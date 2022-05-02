from typing import List
from app.models.image_embeds import User as UserModel
from app.schemas import user as user_schema
from fastapi import  status, HTTPException, Depends, APIRouter
from sqlalchemy.orm import Session
from app.api.deps import get_db 

import glob, torch, clip, shutil
import numpy as np
import pandas as pd

from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

device = 'cuda ' if torch.cude.is_available() else 'cpu'
model, preprocess = clip.load("ViT-B/32", device=device)

print("Loading Model...")


router = APIRouter()

print('Loading Model...')

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


def create_image_embeddings(image_paths, db: Session = Depends(get_db)):
	conn = sqlite3.connect(DATABASE)
	for img_path in image_paths:
		processed_image = preprocess_image(img_path)
		with torch.no_grad():
			embed = model.encode_image(processed_image).detach().numpy()
		embed = embed.tostring()

		## Insert data into sqlite3
		query = "INSERT INTO image_embeds(image_path, embedding) VALUES(?, ?);"
		conn.execute(query, (img_path, embed))
		conn.commit()
		print('Inserted', img_path)
	return 'success'

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
	
	## Calculate cos sim for all images wrt to text
	df['sim'] = df['embedding'].apply(lambda x: cal_sim(text_embed, x))
	
	df = df.sort_values(by=['sim'], ascending=False)
	return df

def image_images_similarity(img_path, df):
	## Preprocess image
	processed_image = preprocess_image(img_path)
	with torch.no_grad():
		image_embed = model.encode_image(processed_image).detach().numpy()

	df['sim'] = df['embedding'].apply(lambda x: cal_sim(image_embed, x))

	df = df.sort_values(by=['sim'], ascending=False)
	return df
	
def get_image_data():
	# conn = sqlite3.connect(DATABASE)
	curr = conn.cursor()

	query = "SELECT image_path, embedding FROM image_embeds;"
	curr.execute(query)

	rows = curr.fetchall()
	return rows

def get_image_data_df():
	# con = sqlite3.connect(DATABASE)
	df = pd.read_sql("SELECT image_path, embedding FROM image_embeds;", con)
	return df

