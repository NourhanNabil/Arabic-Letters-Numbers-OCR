# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app 

# Copy the requirements file into the container at /app
COPY requirements.txt . 

# Install Python dependencies
RUN pip install -r requirements.txt 

# Copy the application code into the container at /app
COPY . .

# Expose port 5000 to the outside world
EXPOSE 5000 

# Command to run the Python script
CMD ["python", "app.py"]