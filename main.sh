#!/bin/bash
nohup /u01/aipoc/miniconda/envs/no.1-sql-assist/bin/python /u01/aipoc/No.1-SQL-Assist/main.py --host 0.0.0.0 > /u01/aipoc/No.1-SQL-Assist/main.log 2>&1 &
exit 0