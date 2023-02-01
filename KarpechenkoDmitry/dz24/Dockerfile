FROM alpine:latest

RUN apk add --no-cache python3-dev \
    && python3 -m ensurepip \
    && pip3 install --upgrade pip

RUN apk add --update --no-cache py3-numpy
RUN apk add htop

WORKDIR /app

COPY . /app

RUN pip3 --no-cache-dir install -r requirements.txt

EXPOSE 5000

CMD ["python3", "app.py"]
