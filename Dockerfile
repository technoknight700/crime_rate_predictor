FROM python:3.10-slim

# system dependencies for GeoPandas / GDAL
RUN apt-get update && \
    apt-get install -y gdal-bin libgdal-dev && \
    apt-get clean

# set GDAL env
ENV GDAL_DATA=/usr/share/gdal
ENV PROJ_LIB=/usr/share/proj

WORKDIR /app

COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["python", "app.py"]
