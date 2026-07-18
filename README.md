# No.1-SQL-Assist
This project is designed to help developers easily generate SQL queries and deepen their understanding of SQL.

# Deploy to OCI

  Click [![Deploy to Oracle Cloud](https://oci-resourcemanager-plugin.plugins.oci.oraclecloud.com/latest/deploy-to-oracle-cloud.svg)](https://cloud.oracle.com/resourcemanager/stacks/create?region=ap-osaka-1&zipUrl=https://github.com/engchina/No.1-SQL-Assist/releases/download/v0.1.10/v0.1.10.zip)


## Network Notes

- The application serves plain HTTP directly from Gradio. The Resource Manager default is TCP port 8080; set `Application port` to 80 or another port when deploying. No nginx proxy or TLS certificate is required.
- Public and private Compute subnets are supported. Public IP assignment and the displayed access IP are selected automatically from the subnet configuration.
- Allow the selected application port in the Compute subnet security list or NSG. For a private subnet, provide NAT access for installation and private network connectivity for application and SSH access.
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
python main.py
```

Port 8080 is the default. Use `python main.py --port 80` or set `Application port` in Resource Manager to override it. Ports below 1024 require root privileges or an equivalent low-port capability.
