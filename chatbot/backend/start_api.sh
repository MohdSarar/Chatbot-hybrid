#!/bin/bash

echo "Lancement du serveur FastAPI..."
uvicorn app.main:app --reload
