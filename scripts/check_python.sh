#!/usr/bin/env bash
# Static checks on Python files changed relative to a base ref (default: main).
# Never runs the analysis: it only byte-compiles and runs pyflakes.
#
# Usage (from anywhere in the repo):
#   scripts/check_python.sh            # compare against main
#   scripts/check_python.sh <base-ref> # compare against another ref
#
# Reports syntax errors, and pyflakes warnings that are NEW compared with the
# base version of each file (pre-existing warnings are ignored). Exit code 1 if
# anything new is found.

base=${1:-main}
cd "$(git rev-parse --show-toplevel)" || exit 2

if ! python3 -m pyflakes --version > /dev/null 2>&1; then
    echo "pyflakes not installed; install it with: python3 -m pip install pyflakes" >&2
    exit 2
fi

files=$(git diff --name-only --diff-filter=AM "$base" -- '*.py'; git ls-files --others --exclude-standard -- '*.py')
files=$(echo "$files" | grep -v '^alian/sandbox/' | sort -u)
if [ -z "$files" ]; then
    echo "No changed Python files relative to $base."
    exit 0
fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
status=0

for f in $files; do
    [ -f "$f" ] || continue
    if ! python3 -m py_compile "$f" 2> "$tmp/compile.err"; then
        echo "SYNTAX ERROR in $f:"
        cat "$tmp/compile.err"
        status=1
        continue
    fi

    # Pyflakes messages with "file:line:col:" stripped, so moved lines still match.
    python3 -m pyflakes "$f" 2>&1 | sed -E 's/^[^:]+:[0-9]+:[0-9]+:? //' | sort > "$tmp/new.txt"
    if git cat-file -e "$base:$f" 2> /dev/null; then
        git show "$base:$f" > "$tmp/base.py"
        python3 -m pyflakes "$tmp/base.py" 2>&1 | sed -E 's/^[^:]+:[0-9]+:[0-9]+:? //' | sort > "$tmp/old.txt"
    else
        : > "$tmp/old.txt"
    fi

    added=$(comm -13 "$tmp/old.txt" "$tmp/new.txt")
    if [ -n "$added" ]; then
        echo "New pyflakes warnings in $f:"
        # Show the full current messages (with line numbers) for the new ones.
        python3 -m pyflakes "$f" 2>&1 | grep -F -f <(echo "$added") | sed 's/^/  /'
        status=1
    else
        echo "OK  $f"
    fi
done

exit $status
