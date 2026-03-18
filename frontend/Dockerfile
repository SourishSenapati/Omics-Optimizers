# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements from the root
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the frontend app and the payload
COPY frontend/app.py ./frontend/
COPY omics_intelligence_payload.json .

# Expose the port Streamlit runs on
EXPOSE 8501

# Start the application
CMD ["streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
