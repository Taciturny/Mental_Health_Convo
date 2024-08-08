FROM python:3.10-slim

# Set the working directory
WORKDIR /Mental_Hea

# Copy the requirements file and install dependencies
COPY Mental_Hea/app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY Mental_Hea/app/ .

# Set the command to run your application
CMD ["python", "main.py"]
