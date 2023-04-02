FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
RUN apt-get update && apt-get install -y python3-pip
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
ADD requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

ADD 20B_tokenizer.json ./
ADD app.py ./

CMD python3 app.py
