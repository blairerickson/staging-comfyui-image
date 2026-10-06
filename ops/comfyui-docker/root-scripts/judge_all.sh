#!/bin/bash
for f in /root/m38/*.png; do
  echo "===== $(basename $f)";
  /usr/bin/python3 /root/judge_box.py "$f" | tr -d '\n' | tail -c 900;
  echo;
done
