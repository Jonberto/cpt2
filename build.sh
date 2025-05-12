set -e

mkdir -p bin

gcc -O2 src/main.c -o bin/main -lm
./bin/main