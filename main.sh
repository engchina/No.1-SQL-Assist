#!/bin/bash

PROJECT_DIR="/u01/aipoc/No.1-SQL-Assist"
PYTHON_BIN="/u01/aipoc/miniconda/envs/no.1-sql-assist/bin/python"

cd "$PROJECT_DIR" || exit 1
source "$PROJECT_DIR/application_port.sh" || exit 1
APPLICATION_PORT="$(resolve_application_port)" || exit 1
nohup "$PYTHON_BIN" main.py --host 0.0.0.0 --port "$APPLICATION_PORT" > "$PROJECT_DIR/main.log" 2>&1 &
exit 0
