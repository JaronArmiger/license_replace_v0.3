# Define function directory
ARG FUNCTION_DIR="/function"

FROM python:3.9.6-slim

# set environment variables
ENV PYTHONUNBUFFERED 1
ENV USE_NNPACK=0
ENV MPLCONFIGDIR="/tmp/matplotlib"
ENV YOLO_CONFIG_DIR="/tmp/yolo"

# Update Dependencies
RUN apt-get update -y; 
RUN apt-get upgrade -y;

# gcc compiler and opencv prerequisites
RUN apt-get -y install git build-essential libglib2.0-0 libsm6 libxext6 libxrender-dev wget cmake ffmpeg
RUN pip3 install --upgrade pip

# Include global arg in this stage of the build
ARG FUNCTION_DIR

# adding requirments to current directory
COPY requirements.txt ${FUNCTION_DIR}/requirements.txt
RUN pip install --target ${FUNCTION_DIR} -r ${FUNCTION_DIR}/requirements.txt --no-cache-dir --compile

COPY . ${FUNCTION_DIR}

WORKDIR ${FUNCTION_DIR}

ENTRYPOINT [ "/usr/local/bin/python", "-m", "awslambdaric" ]
CMD [ "lambda_function.handler" ]