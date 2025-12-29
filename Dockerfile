# -------------------------------
# 1. Base Image (Ubuntu)
# -------------------------------
FROM ubuntu:22.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# -------------------------------
# 2. Install dependencies
# -------------------------------
RUN apt-get install -y \
    curl \
    wget \
    git \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3 \
    python3-pip \
    python3-dev

# -------------------------------
# 3. Install Ollama (Linux)
# -------------------------------
RUN curl -fsSL https://ollama.com/install.sh | sh

# Add Ollama to PATH
ENV PATH="/usr/local/bin:${PATH}"

# Start Ollama service
RUN ollama serve & \
    sleep 5

# -------------------------------
# 4. Pull required Ollama models
# -------------------------------
RUN ollama pull qwen3-vl:latest
RUN ollama pull deepseek-coder-v2:16b
RUN ollama pull deepseek-r1:7b

# -------------------------------
# 5. Install Anaconda
# -------------------------------
WORKDIR /opt

RUN wget https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh -O anaconda.sh && \
    bash anaconda.sh -b -p /opt/anaconda && \
    rm anaconda.sh

ENV PATH="/opt/anaconda/bin:${PATH}"

# -------------------------------
# 6. Create Conda Environment INSIDE PROJECT FOLDER
# -------------------------------
WORKDIR /app

RUN conda create -y --prefix /app/automation_env python=3.10

# Add conda env to PATH
ENV PATH="/Complete_product/automation_env/bin:${PATH}"


# -------------------------------
# 7. Install Python Requirements
# -------------------------------
COPY requirements.txt /app/requirements.txt

RUN conda activate automation_env && \
    pip install --upgrade pip && \
    pip install -r /app/requirements.txt

# -------------------------------
# 8. Copy application source
# -------------------------------
WORKDIR /app
COPY . /app

# -------------------------------
# 9. Expose Streamlit port
# -------------------------------
EXPOSE 8501

# -------------------------------
# 10. Default startup command
# -------------------------------
CMD bash -lc "source activate automation_env && ollama serve & sleep 5 && streamlit run app.py --server.address=0.0.0.0"
