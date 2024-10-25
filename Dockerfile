# Use the official Jupyter base image
FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    python3-setuptools

WORKDIR /home

COPY . /home

RUN pip install --upgrade setuptools pip
RUN pip cache purge
RUN pip install --no-cache-dir -e .
RUN pip install jupyter

# Expose the port for Jupyter Notebook
EXPOSE 8888

# Start Jupyter Notebook
CMD ["jupyter-notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]