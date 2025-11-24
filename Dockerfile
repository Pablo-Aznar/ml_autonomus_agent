# Dockerfile para Hugging Face Spaces (2025)
FROM python:3.10-slim

# Instala dependencias del sistema (para ydata-profiling y matplotlib)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Crea usuario no-root (obligatorio en HF Spaces)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /code

# Copia y instala requirements
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copia el resto del código
COPY --chown=user . .

# Expone puerto 7860 (estándar de HF Spaces)
EXPOSE 7860

# Comando de inicio: uvicorn en puerto 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]