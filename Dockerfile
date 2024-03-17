FROM python:3.10.12-slim

ENV PORT=5000
EXPOSE 5000

# --- start to install backend-end stuff
RUN mkdir -p /app
RUN mkdir -p /assets
WORKDIR /app

# --- Install Python requirements.
RUN apt-get update
RUN apt-get install -y python3
RUN apt-get install -y python3-pip

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# --- Copy project files
COPY ["app.py", "./"]
COPY ["assets", "./assets/"]
COPY ["constants.py", "./"]
COPY ["helper_functions.py", "./"]

# --- Start server
CMD gunicorn app:app --bind 0.0.0.0:$PORT --timeout=600 --threads=10