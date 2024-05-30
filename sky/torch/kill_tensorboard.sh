lsof -i:6006 | grep python3 | awk '{print $2}' | xargs kill -9
rm data/event*
