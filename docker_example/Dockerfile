FROM python:3.10-slim

# Install NumPy
RUN pip install numpy

# Create directory for mounting
WORKDIR /app

# Default entrypoint
ENTRYPOINT ["/app/start.sh"]
