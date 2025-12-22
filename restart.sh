#!/bin/bash
lsof -ti:8080 | xargs kill -9 2>/dev/null || true
/bin/bash /u01/aipoc/No.1-SQL-Assist/main.sh
exit 0
