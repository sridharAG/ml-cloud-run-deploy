FROM python:3.8-slim-buster

WORKDIR /app

COPY ./requirements.txt /app/
COPY ./train.py /app/
COPY ./deploy.py /app/

RUN pip install --no-cache-dir --upgrade -r requirements.txt

RUN python train.py
#COPY ./model.joblib /app/

EXPOSE 8080

CMD ["uvicorn", "deploy:app", "--host", "0.0.0.0", "--port", "8080"]

#RUN mkdir -p data model