from email.mime import application, image
import random
import string
from typing import List

from annoy import AnnoyIndex
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
ANNOY_INDEX_FILE = 'annoy_indexes.ann'
NUM_OF_RESULTS_TO_SHOW = 5
NUM_OF_TREES_TO_BUILD = 5
INDEX_METRIC = 'angular'
embed_length = 512
annoy_indexer = AnnoyIndex(embed_length, INDEX_METRIC)
ANNOY_INDEX = 0




print("Loading Model...")

router = APIRouter()

 

# Image & Text Functions
def preprocess_image(image_path):
	image = Image.open(image_path)
	image = preprocess(image).unsqueeze(0).to(device)
	return image


def preprocess_text(text):
	return clip.tokenize(text).to(device)


def store_uploaded_images(uploaded_files):
	image_paths = []
	for uploaded_file in uploaded_files:
		rand_post_text = ''.join(random.choices(string.ascii_uppercase+string.digits, k = 10))
		fname_split = uploaded_file.filename.split('.')
		fname = fname_split[0] + rand_post_text + "." + fname_split[1]
		file_location = f"{CONTENT_STORE}/{fname}"

		with open(file_location, "wb+") as file_object:
			file_object.write(uploaded_file.file.read())

		image_paths.append(file_location)
	return image_paths


def create_embeddings(db, image_paths):
	global ANNOY_INDEX
	embed_template = {
          'image_path': str,
          'embedding': str
          }

	for img_path in image_paths:
          
		processed_image = preprocess_image(img_path)
		with torch.no_grad():
			embed = model.encode_image(processed_image).detach().numpy()
		embed = embed.tostring()
		
		embed_template['annoy_index']
		embed_template['embedding'] = embed
		embed_template['image_path'] = img_path

		new_embed = ImageEmbedModel(**embed_template)
		embed_template.clear()

		db.add(new_embed)
		db.commit()	
		db.refresh(new_embed)

		annoy_indexer.add_item(ANNOY_INDEX, embed[0])
		ANNOY_INDEX +=1
		print('INSERTED', image_paths)



	return True


def update_annoy_index():
	global ANNOY_INDEX
	query = "SELECT id FROM image_embeds ORDER BY id DESC LIMIT 1;"
	rows = exec_query(query)
	if rows==[]:
		ANNOY_INDEX = 0
	else:
		ANNOY_INDEX = int(rows[0][0]) + 1
	print('Updated Annoy Index as:', ANNOY_INDEX)


def create_image_embeddings(db, uploaded_files):

	image_paths = store_uploaded_images(uploaded_files)

	for image_path in image_paths:
		create_embedding(db, image_path)

	annoy_indexer.build(NUM_OF_TREES_TO_BUILD)
	annoy_indexer.save(ANNOY_INDEX_FILE)

	update_annoy_index()
	return 'Embeddings Created !'


@router.get('/')
def search_text(db: Session = Depends(get_db)):
     return {'msg': 'Searching with text'}


def feat_to_32(feat):
	print('\n\n\n\n THIS IS FIRSTTTT \n\n\n\n\n\n' )
	# print(np.frombuffer(feat.encode(), dtype=np.float32))

	# print(vector_bytes)
	vector_bytes_str = feat
	vector_bytes_str_enc = vector_bytes_str.encode()
	bytes_np_dec = vector_bytes_str_enc.decode('unicode-escape').encode('ISO-8859-1')[2:-1]
	

	np.frombuffer(bytes_np_dec, dtype=np.int64)

	return 0


def cal_sim(feat1, feat2):
	new_arr = feat2.encode()
	img_embed = np.frombuffer(feat2.encode(), dtype=np.float16)
	print("size beforee", len(img_embed))
	# img_embed = img_embed.reshape((1, img_embed.shape[0]))
	print('Its sizeee afterr', len(img_embed))
	# sim = cosine_similarity(feat1, img_embed)
	# print("DAN DAN DUUNNNNNNN", sim)
	# return sim[0][0]
	return ['0']


def text_images_similarity(text, df):

	processed_text = preprocess_text([text])
	
	with torch.no_grad():
		text_embed = model.encode_text(processed_text)
	

	
	# print(f"Heyy and ummm {df['embedding']}")
	## Calculate cos sim for all images wrt to text
	# df['sim'] = df['embedding'].apply(lambda x: cal_sim(text_embed, x))

	# application_x = lambda x: cal_sim(text_embed, x) # THIS IS WHERE THE BUGG IS AHHHH

	# print('THE SIMM WORKSS ahhhh', df['embedding'].apply(application_x))
	
	 
	df['embedding'].apply(lambda x: feat_to_32(x))
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
	
	image_paths = glob.glob(CONTENT_STORE+'/*.jpeg') + glob.glob(CONTENT_STORE+'/*.jpg') + glob.glob(CONTENT_STORE+'/*.png')
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