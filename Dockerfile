FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libgl1 poppler-utils libgthread-2.0-0 tesseract-ocr tesseract-ocr-eng

ENV SENTENCE_TRANSFORMERS_HOME=/app/model_cache
RUN mkdir -p $SENTENCE_TRANSFORMERS_HOME

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data to a local directory
RUN python -m nltk.downloader -d /app/nltk_data punkt averaged_perceptron_tagger

# Set the NLTK_DATA environment variable
ENV NLTK_DATA=/app/nltk_data

# Pre-download the sentence-transformer model
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY . .

CMD ["python", "main.py"]