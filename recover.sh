#!/usr/bin/env bash

headcommit="$(git log --format=format:%H)"
headcommitobject=".git/objects/${headcommit:0:2}${headcommit:2}"
mkdir recovering_lost_files
find .git/objects/ -type f -newer "$headcommitobject"|while read -r path
do
    obj="${path#.git/objects/}"
    obj="${obj/\/}"
    git cat-file -p $obj > recovering_lost_files/$obj
done