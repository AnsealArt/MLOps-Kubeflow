FROM python:3.7-slim

RUN pip install -r requirements.txt

COPY deploy_model.py ./deploy_model.py

ENTRYPOINT ["python", "deploy_model.py"]
