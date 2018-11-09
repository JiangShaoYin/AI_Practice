#!/bin/bash

read -p "input your filename:" fileuser

filename=${fileuser:-"filename"}
date1=$(date --date='1 days ago' +%Y%m%d)
file1=${filename}${date1}

touch "$file1"
