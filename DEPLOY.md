# Complaint Detection System Deployment Guide

This document provides instructions for deploying the Complaint Detection System using Docker.

## Prerequisites

- Docker Engine installed (version 19.03.0+)
- Docker Compose installed (version 1.27.0+)
- At least 4GB of RAM and 2 CPU cores
- Internet connection (for pulling Docker images)

## Quick Deployment

The easiest way to deploy the system is using the provided deployment script:

```bash
# Make the script executable
chmod +x deploy.sh

# Run the deployment script
./deploy.sh
```

The script will:
1. Check for Docker and Docker Compose
2. Create a `.env` file with default values if not present
3. Build and start the Docker containers
4. Display access information

## Manual Deployment

If you prefer to deploy manually or the script doesn't work for your environment:

1. Create a `.env` file with the following contents:
```
TOKEN_SECRET_KEY=CHANGE_ME_IN_PRODUCTION
ENABLE_AUTH=true
ADMIN_USERNAME=admin
ADMIN_PASSWORD=adminpassword
MAX_CONTENT_LENGTH=10000
```

2. Build and start the containers:
```bash
docker-compose up --build -d
```

3. Verify the service is running:
```bash
docker-compose ps
```

## Accessing the Application

Once deployed, you can access:

- UI Application: http://localhost:8000
- API Documentation: http://localhost:8000/docs

Default login credentials:
- Username: admin
- Password: adminpassword

> **IMPORTANT:** For production use, change the default credentials and TOKEN_SECRET_KEY in the .env file.

## Stopping the Service

To stop the service:

```bash
docker-compose down
```

## Updating the Service

When new updates are available:

1. Pull the latest code changes
2. Rebuild and restart the containers:
```bash
docker-compose down
docker-compose up --build -d
```

## Troubleshooting

1. **Service not accessible:**
   - Check if containers are running: `docker-compose ps`
   - Check logs: `docker-compose logs`

2. **Login issues:**
   - Verify credentials in the `.env` file

3. **Performance problems:**
   - Check system resources: `docker stats`
   - Consider increasing resources allocated to Docker

4. **Container exit unexpectedly:**
   - Check logs for errors: `docker-compose logs complaint-detection-app`

## Production Considerations

For production deployment:

1. Use a proper secret key for TOKEN_SECRET_KEY
2. Change the default admin credentials
3. Consider setting up HTTPS with a reverse proxy (Nginx, Traefik, etc.)
4. Set up monitoring and logging solutions
5. Configure regular backups of any persistent data

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| TOKEN_SECRET_KEY | Secret key for JWT tokens | CHANGE_ME_IN_PRODUCTION |
| ENABLE_AUTH | Enable authentication | true |
| ADMIN_USERNAME | Admin username | admin |
| ADMIN_PASSWORD | Admin password | adminpassword |
| MAX_CONTENT_LENGTH | Max content size in bytes | 10000 |
| PORT | Port for the service | 8000 |
| HOST | Host address binding | 0.0.0.0 |
| MODEL_PATH | Path to ML models | /app/models | 