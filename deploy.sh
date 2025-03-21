#!/bin/bash

# Simple deployment script for Complaint Detection System

# Set terminal colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Complaint Detection System - Deployment Script${NC}"
echo -e "----------------------------------------------"

# Check for Docker and Docker Compose
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed. Please install Docker first.${NC}"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Error: Docker Compose is not installed. Please install Docker Compose first.${NC}"
    exit 1
fi

# Check if .env file exists, create if not
if [ ! -f .env ]; then
    echo -e "${YELLOW}Creating .env file with default values...${NC}"
    cat > .env << EOL
TOKEN_SECRET_KEY=CHANGE_ME_IN_PRODUCTION
ENABLE_AUTH=true
ADMIN_USERNAME=admin
ADMIN_PASSWORD=adminpassword
MAX_CONTENT_LENGTH=10000
EOL
    echo -e "${GREEN}Created .env file. Please review and edit as needed.${NC}"
else
    echo -e "${GREEN}Found existing .env file.${NC}"
fi

# Build and start containers
echo -e "${YELLOW}Building and starting containers...${NC}"
docker-compose up --build -d

# Check if startup was successful
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Deployment successful!${NC}"
    echo -e "\nAccess the application at: ${GREEN}http://localhost:8000${NC}"
    echo -e "Default login credentials: ${GREEN}admin / adminpassword${NC}"
    echo -e "\n${YELLOW}Important:${NC} For production use, please change the default credentials in the .env file."
else
    echo -e "${RED}Deployment failed. Please check the error messages above.${NC}"
fi 