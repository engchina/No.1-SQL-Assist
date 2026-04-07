#!/bin/bash

PROJECT_DIR="/u01/aipoc/No.1-SQL-Assist"
PYTHON_BIN="/u01/aipoc/miniconda/envs/no.1-sql-assist/bin/python"

cd "$PROJECT_DIR" || exit 1
nohup "$PYTHON_BIN" main.py --host 0.0.0.0 > "$PROJECT_DIR/main.log" 2>&1 &
exit 0
