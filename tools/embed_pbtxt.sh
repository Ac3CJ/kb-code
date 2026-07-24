#!/bin/bash
# Converts a .pbtxt file into a C++ header containing the text as a string literal.
# Usage: embed_pbtxt.sh <input.pbtxt> <output.h> <variable_name>
set -euo pipefail

INFILE="$1"
OUTFILE="$2"
VARNAME="$3"

if [ ! -f "$INFILE" ]; then
    echo "Error: input file '$INFILE' not found." >&2
    exit 1
fi

mkdir -p "$(dirname "$OUTFILE")"

# Escape the file contents into a C++ string literal.
# Use hex escapes for safety (avoids issues with unprintable chars,
# backslashes, quotes, etc.)
{
    printf "// Auto-generated from %s. Do not edit.\n" "$(basename "$INFILE")"
    printf "#pragma once\n"
    printf "#include <string>\n\n"
    printf "inline constexpr const char* %s = R\"MP_GRAPH(\n" "$VARNAME"
    cat "$INFILE"
    printf ")MP_GRAPH\";\n"
} > "$OUTFILE"

echo "Generated $OUTFILE"