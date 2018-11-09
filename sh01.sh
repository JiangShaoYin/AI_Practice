#!/bin/bash
PATH=/bin:/sbin:/usr/bin
export PATH
git add .
git commit -m "auto"
git push origin master
echo "git done!"
exit 0
