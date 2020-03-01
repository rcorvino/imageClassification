FROM tensorflow/tensorflow:nightly-py3

WORKDIR /root

ADD . /root

RUN pip install -r /root/requirements.txt

