#!/bin/bash  
message="auto-commit from $USER@$(hostname -s) on $(date)"
git add .
git commit -m"$message"
git push -u origin master
