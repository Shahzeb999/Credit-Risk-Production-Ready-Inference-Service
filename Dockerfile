# use a slim Python image (smaller, faster for production)
FROM python:3.10-slim

# set the working directory in the container
WORKDIR /app

# copy the requirements.txt file into the container
COPY requirements.txt .

# install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# copy the application and artifacts (model + config)
COPY app/ ./app/
COPY artifacts/ ./artifacts/

# expose the port 8000
EXPOSE 8000

# run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]