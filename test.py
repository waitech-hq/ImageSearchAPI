from os import pathsep


paths = [
'/Users/elvischege/Desktop/Python/FastAPI/ImageSearchEngine/./app/static/food_one.jpeg',
'/Users/elvischege/Desktop/Python/FastAPI/ImageSearchEngine/./app/static/dog_drawC1VV89NVSS.jpeg',
'/Users/elvischege/Desktop/Python/FastAPI/ImageSearchEngine/./app/static/dog.jpeg',
'/Users/elvischege/Desktop/Python/FastAPI/ImageSearchEngine/./app/static/aadhPGTWW562PA.jpeg',
'/Users/elvischege/Desktop/Python/FastAPI/ImageSearchEngine/./app/static/docZEML4SLAR6.jpeg'
]

arr = list()
for i in range(len(paths)):
    arr.append({})

print(arr)
for index, path in enumerate(paths):
    arr[index]['paths'] = path
    print(arr)