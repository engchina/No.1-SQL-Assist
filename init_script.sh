#!/bin/bash
echo "Initializing application setup..."

cd /u01/aipoc

# Install and configure miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /u01/aipoc/miniconda.sh
bash /u01/aipoc/miniconda.sh -b -p /u01/aipoc/miniconda
eval "$(/u01/aipoc/miniconda/bin/conda shell.bash hook)"
/u01/aipoc/miniconda/bin/conda init bash

# Download and configure instantclient
wget https://download.oracle.com/otn_software/linux/instantclient/2380000/instantclient-basic-linux.x64-23.8.0.25.04.zip -O /u01/aipoc/instantclient-basic-linux.x64-23.8.0.25.04.zip
unzip /u01/aipoc/instantclient-basic-linux.x64-23.8.0.25.04.zip -d ./
wget http://ftp.de.debian.org/debian/pool/main/liba/libaio/libaio1_0.3.113-4_amd64.deb
dpkg -i libaio1_0.3.113-4_amd64.deb
sh -c "echo /u01/aipoc/instantclient_23_8 > /etc/ld.so.conf.d/oracle-instantclient.conf"
ldconfig
echo 'export LD_LIBRARY_PATH=/u01/aipoc/instantclient_23_8:$LD_LIBRARY_PATH' >> /etc/profile
source /etc/profile

# Unzip wallet and copy essential file to instantclient
unzip /u01/aipoc/wallet.zip -d ./wallet
cp ./wallet/*  /u01/aipoc/instantclient_23_8/network/admin/

# Move to source directory
cd /u01/aipoc/No.1-SQL-Assist

dos2unix main.cron
crontab main.cron

# Update environment variables
DB_CONNECTION_STRING=$(cat /u01/aipoc/props/db.env)
COMPARTMENT_ID=$(cat /u01/aipoc/props/compartment_id.txt)
cp .env.example .env
DB_CONNECTION_STRING=$(cat /u01/aipoc/props/db.env)
sed -i "s|ORACLE_26AI_CONNECTION_STRING=TODO|ORACLE_26AI_CONNECTION_STRING=$DB_CONNECTION_STRING|g" .env
COMPARTMENT_ID=$(cat /u01/aipoc/props/compartment_id.txt)
sed -i "s|OCI_COMPARTMENT_OCID=TODO|OCI_COMPARTMENT_OCID=$COMPARTMENT_ID|g" .env
ADB_NAME=$(cat /u01/aipoc/props/adb_name.txt 2>/dev/null || true)
if [ -n "$ADB_NAME" ]; then sed -i "s|ADB_NAME=TODO|ADB_NAME=$ADB_NAME|g" .env; fi

# Application setup
EXTERNAL_IP=$(curl -s -m 10 http://whatismyip.akamai.com/)
echo $EXTERNAL_IP

# Accept conda pkgs
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Run application
conda create -n no.1-sql-assist python=3.11 -y
conda activate no.1-sql-assist
pip install -r requirements.txt

chmod +x main.sh
nohup ./main.sh &

echo "Initialization complete."
