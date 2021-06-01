# syntax=docker/dockerfile:1

FROM ubuntu:focal
WORKDIR /lvmdrp
ARG LVMHUB=../

# create python virtual environment
# write to Joel, Jose and Eric and ask for a placeholder

COPY  requirements_ubuntu.txt requirements_ubuntu.txt
RUN apt-get update
RUN apt install -y $(awk '{print $1'} $LVMHUB/lvmdrp/requirements_ubuntu.txt)

RUN apt-get install -y python3 python3-pip git

#COPY requirements.txt requirements.txt
#COPY requirements_doc.txt requirements_doc.txt
#COPY requirements_dev.txt requirements_dev.txt
#RUN pip install -r requirements_dev.txt

COPY . .
RUN bash $LVMHUB/lvmdrp/utils/install.sh
