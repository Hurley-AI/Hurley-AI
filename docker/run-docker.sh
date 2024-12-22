#!/bin/bash

docker build docker/ --tag="hurley-agentic-demo:latest"

if [ $? -eq 0 ]; then
  echo "Docker build successful."
else
  echo "Docker build failed. Please check the messages above. Exiting..."
  exit 4
fi

# remove old container if it exists
docker container inspect hurley-agentic-demo &>/dev/null && docker rm -f hurley-agentic-demo

# Run docker container
docker run -p 8000:8000 --name hurley-agentic-demo hurley-agentic-demo:latest

if [ $? -eq 0 ]; then
  echo "Success! hurley-agentic simple agent is running."
  echo "Go to http://localhost:8001 to access the hurley-agentic simple agent."
else
  echo "Hurley-agentic container failed to start. Please check the messages above."
fi