# No.1-SQL-Assist
This project is designed to help developers easily generate SQL queries and deepen their understanding of SQL.

# Deploy to OCI

  Click [![Deploy to Oracle Cloud](https://oci-resourcemanager-plugin.plugins.oci.oraclecloud.com/latest/deploy-to-oracle-cloud.svg)](https://cloud.oracle.com/resourcemanager/stacks/create?region=ap-osaka-1&zipUrl=https://github.com/engchina/No.1-SQL-Assist/releases/download/v0.1.8/v0.1.8.zip)


## Network Notes

- The application serves plain HTTP directly from Gradio on TCP port 80. No nginx proxy or TLS certificate is required.
- Allow inbound TCP port 80 in the Compute subnet security list or NSG.
- If ADB is deployed with a private endpoint, allow the Compute instance private IP to access TCP port 1522 in the ADB subnet security list or NSG.


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
python main.py --port 8000
```

Port 80 is the deployment default and requires root privileges or an equivalent low-port capability. Port 8000 is recommended for local development.
