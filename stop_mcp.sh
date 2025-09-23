#!/bin/bash

echo "Stopping and cleaning up services..."
# The --remove-orphans flag cleans up any containers for services
# that are no longer defined in the docker-compose.yml file.
docker-compose down --remove-orphans
echo "MCP services stopped and cleaned up."
