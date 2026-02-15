# use a slime Python Image

FROM python:3.10-slim # slim is a smaller image for faster download used for production

# set the working directory in the container
WORKDIR /app

# copy the requirements.txt file into the container
COPY requirements.txt .

# install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# copy the rest of the application code into the container
COPY app/ .

# expose the port 8000
EXPOSE 8000

# run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]