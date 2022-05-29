FROM python:3.7-slim

RUN pip install -r requirements.txt

COPY test_model.py ./test_model.py

ENTRYPOINT ["python", "test_model.py"]
