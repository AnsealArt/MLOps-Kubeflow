FROM python:3.7-slim

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY preprocess_data.py ./preprocess_data.py

ENTRYPOINT ["python", "preprocess_data.py"]
