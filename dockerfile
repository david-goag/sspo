FROM python:3.10.6-buster
COPY requirements.txt /requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
COPY sspo /sspo
COPY setup.py /setup.py
COPY xgb_reg_24_54.pkl /xgb_reg_24_54.pkl
RUN pip install .

CMD uvicorn sspo.api.fast:app --host 0.0.0.0 --port $PORT
