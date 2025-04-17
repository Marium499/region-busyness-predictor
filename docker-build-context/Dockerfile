FROM mcr.microsoft.com/azureml/base

# Install Python and pip
RUN apt update && apt install -y python3 python3-pip

# Create working directory
WORKDIR /app

# Copy and install requirements
COPY requirements.txt .
RUN pip install -r src/requirements.txt