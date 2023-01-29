FROM alpine:latest

RUN apk add --no-cache python3-dev \
    && python3 -m ensurepip \
    && pip3 install --upgrade pip

WORKDIR /app

COPY . /app

RUN pip3 install flask datetime

EXPOSE 5000

CMD ["python3","exp.py"]

