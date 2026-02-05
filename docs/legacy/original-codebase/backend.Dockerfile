FROM python:3.8-slim

RUN apt-get update -y && apt-get install -y git

WORKDIR /app

COPY backend/requirements.txt /app
RUN pip3 install -r requirements.txt

COPY wait-for-it.sh /app
COPY backend /app/

ENV PYTHONPATH /app
ENTRYPOINT ["python3"]
CMD ["rest_api.py"]
