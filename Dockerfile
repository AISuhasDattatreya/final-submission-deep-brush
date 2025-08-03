FROM tensorflow/tensorflow:2.14.0
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Expose Gradio and FastAPI ports
EXPOSE 7860
EXPOSE 8000

CMD ["bash", "-c", "uvicorn app.main:app --host 0.0.0.0 --port 8000 & python app/interface.py"]
