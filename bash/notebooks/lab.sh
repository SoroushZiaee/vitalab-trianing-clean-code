#!/bin/bash

# Session name
# SESSION_NAME="my_jupyter_session"

# Check if the session exists
# if tmux has-session -t $SESSION_NAME 2>/dev/null; then
#     echo "Session $SESSION_NAME exists. Attaching and starting Jupyter Lab..."
#     # Attach to the session and send the command to start Jupyter Lab
#     tmux attach-session -t $SESSION_NAME \; send-keys "jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --ServerApp.token='' --ServerApp.password=''" C-m

#     tmux detach-client
# else
#     echo "Session $SESSION_NAME does not exist. Creating and starting Jupyter Lab..."
#     # Create a new session, start Jupyter Lab, and detach
#     tmux new-session -d -s $SESSION_NAME "jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --ServerApp.token='' --ServerApp.password=''"
#     # Optionally, detach from the session
#     tmux detach-client
# fi

unset XDG_RUNTIME_DIR
jupyter lab --ip=$(hostname -f) --port=8008 --no-browser --allow-root --ServerApp.token='' --ServerApp.password=''
