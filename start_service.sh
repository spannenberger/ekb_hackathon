#!/bin/bash
sh install_models.sh
gunicorn --log-level debug --timeout 360 --bind 0.0.0.0:5000 backend:app