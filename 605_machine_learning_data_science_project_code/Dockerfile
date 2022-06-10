FROM public.ecr.aws/lambda/python:3.8

WORKDIR /build/

ENV PYTHONPATH=/build/

COPY lambda_inference.py /build/
COPY LSTM_price.onnx /build/
COPY requirements.txt /build/

RUN pip install -r requirements.txt

CMD ["lambda_inference.lambda_handler"]

