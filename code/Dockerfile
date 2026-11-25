FROM python:3.12-slim

WORKDIR /app

COPY code ./code
COPY assets ./assets
COPY demo.html index.html ./
COPY chapter-*.html ./

EXPOSE 8000

CMD ["python", "code/serve_pure.py", "--host", "0.0.0.0", "--port", "8000"]
