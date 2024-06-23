#!/bin/bash

# Start the server
python main.py &
  
# Start the signal
python signal_sender.py &
  
# Wait for any process to exit
wait -n
  
# Exit with status of process that exited first
exit $?