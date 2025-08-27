import pickle
from fastapi import FastAPI

app = FastAPI()
xgb_reg = pickle.load(open('xgb_reg.pkl', 'rb'))

@app.get('/')
def index():
    return {'ok': True}
