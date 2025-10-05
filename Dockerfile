# Use official Python 3.10 image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all app files
COPY . /app

# Create Streamlit config directory and file
RUN mkdir -p /app/.streamlit
RUN echo '[server]\n\
port = 8080\n\
address = "0.0.0.0"\n\
headless = true\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
enableWebsocketCompression = false\n\
maxUploadSize = 200\n\
\n\
[browser]\n\
gatherUsageStats = false\n\
serverAddress = "madebymtaher.in"\n\
serverPort = 443' > /app/.streamlit/config.toml

# Expose port 8080
EXPOSE 8080

# Command to run the app
ENTRYPOINT ["streamlit", "run", "main.py"]
