FROM python:3.10.6-buster
COPY requirements.txt /requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
COPY strava-pacing-optimizer /strava-pacing-optimizer
COPY setup.py /setup.py
RUN pip install .

CMD uvicorn strava-pacing-optimizer.api.fast:app --host 0.0.0.0
