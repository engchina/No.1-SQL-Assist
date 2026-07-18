#!/bin/bash

PROJECT_DIR="/u01/aipoc/No.1-SQL-Assist"

cd "$PROJECT_DIR" || exit 1
source "$PROJECT_DIR/application_port.sh" || exit 1
APPLICATION_PORT="$(resolve_application_port)" || exit 1
lsof -ti:"$APPLICATION_PORT" | xargs kill -9 2>/dev/null || true
/bin/bash "$PROJECT_DIR/main.sh"
exit 0
