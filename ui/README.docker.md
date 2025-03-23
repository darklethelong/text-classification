# Complaint Detection UI - Docker Deployment

This document explains how to deploy the Complaint Detection UI frontend that connects to the remote backend service.

## Prerequisites

- Docker and Docker Compose installed
- Access to the backend API at www.backend-complaint.com

## Deploying with Docker Compose

The simplest way to deploy the UI is with Docker Compose:

```bash
cd ui
docker-compose up -d
```

This will build the Docker image and start the container with the correct configuration.

## Environment Variables

The UI container supports the following environment variables:

- `API_URL`: The URL of the backend API service (default: https://www.backend-complaint.com)

You can change the backend URL by modifying the docker-compose.yml file or by setting the environment variable when running the container.

## Manual Docker Deployment

If you prefer to deploy without Docker Compose:

```bash
# Build the image
docker build -t complaint-ui -f ui/Dockerfile .

# Run the container
docker run -d -p 80:80 -e API_URL=https://www.backend-complaint.com --name complaint-ui complaint-ui
```

## Authentication

The UI will use the following authentication flow:

1. Demo authentication: Use username `demo` and password `demo` to login using the hardcoded demo token
2. Token authentication: For actual API authentication, the API uses Bearer token authentication

## Customizing the Backend URL

You can customize the backend URL in several ways:

1. At build time by modifying the `REACT_APP_API_URL` in the Dockerfile
2. At runtime by setting the `API_URL` environment variable
3. By modifying the docker-compose.yml file

## Health Check

Once deployed, you can check if the UI can connect to the backend by:

1. Opening the UI in your browser (http://localhost if deployed locally)
2. Logging in with username `demo` and password `demo`
3. The dashboard will show a connection status indicator 