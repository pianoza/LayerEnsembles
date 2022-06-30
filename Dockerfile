FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1
# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y git vim htop wget python-opencv libgl1-mesa-dev
RUN wget -O /tmp/anaconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py38_4.11.0-Linux-x86_64.sh
RUN chmod +x /tmp/anaconda.sh
RUN sh -c /bin/echo -e "yes\n" | /tmp/anaconda.sh -b -p $HOME/anaconda3
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

#RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
#RUN conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
RUN conda install pytorch torchvision cudatoolkit=11 -c pytorch-nightly

# Change workspace to the mounted directory in the container
WORKDIR /home/kaisar/EuCanImage/Coding/LayerEnsembles

CMD [ "/home/kaisar/EuCanImage/Coding/LayerEnsembles/main.py" ]
ENTRYPOINT [ "python" ]

# Tensorboard
EXPOSE 6006

# docker build -t name:sometag
# docker run -d -v /home/kaisar:/home/kaisar name:sometag
# INTERACTIVE
# docker run -it --shm-size=12GB --gpus all -v /home/kaisar:/home/kaisar --entrypoint /bin/bash le:test
# DIRECTLY RUN
# docker run -it --shm-size=12GB --gpus all -v /home/kaisar:/home/kaisar le:test
# Change the CMD for a file to run
# docker run -it --shm-size=12GB --gpus all -v /home/kaisar:/home/kaisar le:test /home/kaisar/EuCanImage/Coding/LayerEnsembles/main2.py
