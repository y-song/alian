#!/usr/bin/env bash
set -euo pipefail

for x in 9 10 18 45 52 53 54 57 60 68 80 81 82 83 84 85 86 87 88 89 90 91 92 93 95 96 97 98 106 112 118; do
    mkdir -p "/rstorage/youqi/1998266/$x"
    mv -iv "/rstorage/youqi/1998422/$x/out.root" \
           "/rstorage/youqi/1998266/$x/out.root"
done