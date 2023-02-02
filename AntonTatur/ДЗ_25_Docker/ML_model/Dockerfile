FROM python:3.8-slim

COPY . /app
WORKDIR /app
RUN pip install --quiet --no-cache-dir -r requirements.txt
EXPOSE 5000

CMD ["python", "mk_n_flask.py"]