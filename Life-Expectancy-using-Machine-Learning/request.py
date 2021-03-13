import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'Status':1,'Schooling':1,'Income_Comp_Of_Resources':1,'HIV/AIDS':1,'Adult_Mortality':1,'Alcohol':1})

print(r.json())