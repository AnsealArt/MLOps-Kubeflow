FROM python:3.7-slim

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY train_model.py ./train_model.py

ENTRYPOINT ["python", "train_model.py"]
