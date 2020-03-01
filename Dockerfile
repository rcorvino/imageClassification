FROM tensorflow/tensorflow:nightly-py3

WORKDIR /root

ADD ../imageClassification.py /root
ADD ../models /root
ADD ../requirements.txt /root
ADD ../selected /root
ADD ../data /root

RUN pip install -r /root/requirements.txt

