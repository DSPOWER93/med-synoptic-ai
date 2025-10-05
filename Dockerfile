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
enableCORS = false\n\
enableXsrfProtection = false\n\
enableWebsocketCompression = false\n\
address = "0.0.0.0"\n\
port = 8080\n\
headless = true\n\
runOnSave = false\n\
\n\
[browser]\n\
gatherUsageStats = false' > /app/.streamlit/config.toml

# Expose port 8080
EXPOSE 8080

# Command to run the app with proxy-friendly settings
ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8080", "--server.address=0.0.0.0", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
