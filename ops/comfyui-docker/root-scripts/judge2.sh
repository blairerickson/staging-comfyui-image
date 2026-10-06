#!/bin/bash
for f in /root/app_before.png /root/app_after.png; do echo "=== $f"; /usr/bin/python3 /root/judge_box.py "$f"; done
