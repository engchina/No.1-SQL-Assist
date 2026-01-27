# No.1-SQL-Assist
This project is designed to help developers easily generate SQL queries and deepen their understanding of SQL.

# Deploy to OCI

  Click [![Deploy to Oracle Cloud](https://oci-resourcemanager-plugin.plugins.oci.oraclecloud.com/latest/deploy-to-oracle-cloud.svg)](https://cloud.oracle.com/resourcemanager/stacks/create?region=ap-osaka-1&zipUrl=https://github.com/engchina/No.1-SQL-Assist/releases/download/v0.1.3/v0.1.3.zip)


# Local Install

```bash
conda create -n no.1-sql-assist python=3.12 -y
```

```bash
conda activate no.1-sql-assist
```

```
pip install -r requirements.txt
# pip list --format=freeze > requirements.txt
```

```
python main.py
```