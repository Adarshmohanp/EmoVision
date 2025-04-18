#!/bin/bash
echo "Starting server on port $PORT"
gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 0 --access-logfile - --error-logfile - app:app