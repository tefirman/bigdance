
# Using the Python 3.10 base image
FROM python:3.10

# Adding labels for the GitHub Container Registry
LABEL org.opencontainers.image.title="tfdancing"
LABEL org.opencontainers.image.description="Docker container for the use of the 'tfdancing' March Madness simulation module"
LABEL org.opencontainers.image.version="latest"
LABEL org.opencontainers.image.authors="tefirman@gmail.com"
LABEL org.opencontainers.image.url=https://taylorfirman.com/
LABEL org.opencontainers.image.source=https://github.com/tefirman/dancing
LABEL org.opencontainers.image.licenses=MIT

# Installing prerequisite python modules via pip
RUN pip install numpy pandas beautifulsoup4 requests lxml

