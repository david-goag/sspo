FROM python:3.10.6-buster
COPY requirements.txt /requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
COPY sspo /sspo
COPY setup.py /setup.py
RUN pip install .
COPY model_500m_no_power_max.pkl model_500m_no_power_max.pkl
COPY xgb_reg_5_84.pkl xgb_reg_5_84.pkl
CMD uvicorn sspo.api.fast:app --host 0.0.0.0 --port $PORT
