FROM node:18-alpine as build

WORKDIR /app

# Copy package files and install dependencies
COPY ui/package.json ui/package-lock.json ./
RUN npm ci

# Copy source code
COPY ui/public ./public
COPY ui/src ./src

# Set environment variables for production build
# Override the API URL to point to the remote backend
ENV REACT_APP_API_URL=http://localhost:8000
ENV NODE_ENV=production

# Build the app
RUN npm run build

# Production stage
FROM nginx:alpine

# Copy built app from build stage
COPY --from=build /app/build /usr/share/nginx/html

# Copy custom nginx config if needed
# COPY ui/nginx.conf /etc/nginx/conf.d/default.conf

# Create a runtime configuration script to allow env variable substitution at runtime
RUN echo '#!/bin/sh' > /docker-entrypoint.d/40-set-api-url.sh && \
    echo 'API_URL=${API_URL:-http://localhost:8000}' >> /docker-entrypoint.d/40-set-api-url.sh && \
    echo 'echo "window.REACT_APP_API_URL=\"$API_URL\";" > /usr/share/nginx/html/config.js' >> /docker-entrypoint.d/40-set-api-url.sh && \
    chmod +x /docker-entrypoint.d/40-set-api-url.sh

# Create a placeholder config.js file
RUN echo 'window.REACT_APP_API_URL="https://www.backend-complaint.com";' > /usr/share/nginx/html/config.js

# Expose port
EXPOSE 80

# No need to specify CMD as nginx image has its own entry point 