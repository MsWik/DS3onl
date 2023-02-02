FROM ubuntu:latest

RUN apt-get update
RUN apt-get install -y samba nginx openssl git wget curl python3-dev
RUN apt-get install -y python3-pip


WORKDIR /app

COPY . /app

RUN pip3 --no-cache-dir install -r requirements.txt

EXPOSE 5000

CMD ["python3", "app.py"]