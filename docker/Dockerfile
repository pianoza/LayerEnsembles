FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y git vim htop wget python-opencv libgl1-mesa-dev
# RUN wget -O /tmp/anaconda.sh https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh
RUN wget -O /tmp/anaconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py38_4.11.0-Linux-x86_64.sh
RUN chmod +x /tmp/anaconda.sh
RUN sh -c /bin/echo -e "yes\n" | /tmp/anaconda.sh -b -p $HOME/anaconda3
#RUN conda init
#RUN conda activate base
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt


# CMD [ "/home/kaisar/projects/uncertainty-quantification/methods/rocket_launch.py" ]
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
