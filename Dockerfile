FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libheif1 \
    libde265-0 \
    imagemagick \
    libmagickcore-6.q16-6-extra \
    && rm -rf /var/lib/apt/lists/* \
    && sed -i 's/rights="none" pattern="HEIC"/rights="read|write" pattern="HEIC"/' /etc/ImageMagick-6/policy.xml 2>/dev/null || true \
    && sed -i 's/<policy domain="coder" rights="none" pattern="PDF"/<policy domain="coder" rights="read|write" pattern="PDF"/' /etc/ImageMagick-6/policy.xml 2>/dev/null || true

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
