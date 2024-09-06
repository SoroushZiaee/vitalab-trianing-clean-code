#!/bin/bash

# Default log directory
LOG_DIR="./logs"

# Check if an argument is provided
if [ "$#" -eq 1 ]; then
    LOG_DIR=$1
fi

# Start TensorBoard
echo "Starting TensorBoard with log directory: $LOG_DIR"
tensorboard --logdir="$LOG_DIR" --bind_all --port=8181 --load_fast=true