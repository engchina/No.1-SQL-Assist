#!/bin/bash
set -euo pipefail

# Redirect all output to log file
exec > >(tee -a /var/log/init_script.log) 2>&1

echo "アプリケーションのセットアップを初期化中..."

# Configuration
INSTALL_DIR="/u01/aipoc"
INSTANTCLIENT_VERSION="23.26.0.0.0"
INSTANTCLIENT_ZIP="instantclient-basic-linux.x64-${INSTANTCLIENT_VERSION}.zip"
INSTANTCLIENT_URL="https://download.oracle.com/otn_software/linux/instantclient/2326000/${INSTANTCLIENT_ZIP}"
INSTANTCLIENT_SQLPLUS_ZIP="instantclient-sqlplus-linux.x64-${INSTANTCLIENT_VERSION}.zip"
INSTANTCLIENT_SQLPLUS_URL="https://download.oracle.com/otn_software/linux/instantclient/2326000/${INSTANTCLIENT_SQLPLUS_ZIP}"
LIBAIO_DEB="libaio1_0.3.113-4_amd64.deb"
LIBAIO_URL="http://ftp.de.debian.org/debian/pool/main/liba/libaio/${LIBAIO_DEB}"
INSTANTCLIENT_DIR="${INSTALL_DIR}/instantclient_23_26"

# Helper function for retrying commands
retry_command() {
    local max_attempts=5
    local timeout=10
    local attempt=1
    local exit_code=0

    while [ $attempt -le $max_attempts ]; do
        echo "Attempt $attempt of $max_attempts: $@"
        "$@" && return 0
        exit_code=$?
        echo "Command failed with exit code $exit_code. Retrying in $timeout seconds..."
        sleep $timeout
        attempt=$((attempt + 1))
        timeout=$((timeout * 2))
    done

    echo "Command failed after $max_attempts attempts."
    return $exit_code
}

cd "$INSTALL_DIR"

# Install and configure miniconda
echo "Minicondaをインストール中..."
if [ ! -d "${INSTALL_DIR}/miniconda" ]; then
    retry_command wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /u01/aipoc/miniconda.sh
    bash /u01/aipoc/miniconda.sh -b -p /u01/aipoc/miniconda
    eval "$(/u01/aipoc/miniconda/bin/conda shell.bash hook)"
    /u01/aipoc/miniconda/bin/conda init bash
else
    echo "Minicondaは既にインストールされています。"
    eval "$(/u01/aipoc/miniconda/bin/conda shell.bash hook)"
fi

# Install Oracle Instant Client 23.26
echo "Oracle Instant Client 23.26をインストール中..."
if [ ! -d "${INSTANTCLIENT_DIR}" ]; then
    if [ ! -f "$INSTANTCLIENT_ZIP" ]; then
        retry_command wget "$INSTANTCLIENT_URL" -O "$INSTANTCLIENT_ZIP"
    fi
    unzip -o "$INSTANTCLIENT_ZIP" -d ./
    
    # Install SQL*Plus
    echo "SQL*Plusをインストール中..."
    if [ ! -f "$INSTANTCLIENT_SQLPLUS_ZIP" ]; then
        retry_command wget "$INSTANTCLIENT_SQLPLUS_URL" -O "$INSTANTCLIENT_SQLPLUS_ZIP"
    fi
    unzip -o "$INSTANTCLIENT_SQLPLUS_ZIP" -d ./

    if [ ! -f "$LIBAIO_DEB" ]; then
        retry_command wget "$LIBAIO_URL"
    fi
    dpkg -i "$LIBAIO_DEB" || apt-get install -f -y
    
    sh -c "echo ${INSTANTCLIENT_DIR} > /etc/ld.so.conf.d/oracle-instantclient.conf"
    ldconfig
    
    if ! grep -q "LD_LIBRARY_PATH=${INSTANTCLIENT_DIR}" /etc/profile; then
        echo "export LD_LIBRARY_PATH=${INSTANTCLIENT_DIR}:\$LD_LIBRARY_PATH" >> /etc/profile
        echo "export PATH=${INSTANTCLIENT_DIR}:\$PATH" >> /etc/profile
    fi
else
    echo "Oracle Instant Client 23.26は既にインストールされています。"
fi

# Safe sourcing of profile
set +eu
source /etc/profile
set -eu
# Explicitly export in case sourcing failed or didn't pick up immediately
export LD_LIBRARY_PATH="${INSTANTCLIENT_DIR}:${LD_LIBRARY_PATH:-}"
export PATH="${INSTANTCLIENT_DIR}:$PATH"

# Verify sqlplus installation
if command -v sqlplus >/dev/null 2>&1; then
    echo "SQL*Plusのインストール検証が成功しました"
else
    echo "エラー: SQL*Plusのインストール検証に失敗しました"
    exit 1
fi

# Move to source directory
echo "No.1-SQL-Assistプロジェクトをセットアップ中..."
cd /u01/aipoc/No.1-SQL-Assist

dos2unix main.cron
crontab main.cron

# Update environment variables
echo "環境変数を設定中..."
cp .env.example .env

# Check for property files before reading
if [ -f "/u01/aipoc/props/db.env" ]; then
    DB_CONNECTION_STRING=$(cat /u01/aipoc/props/db.env)
    sed -i "s|ORACLE_26AI_CONNECTION_STRING=TODO|ORACLE_26AI_CONNECTION_STRING=$DB_CONNECTION_STRING|g" .env
else
    echo "警告: /u01/aipoc/props/db.env が見つかりません！"
fi

if [ -f "/u01/aipoc/props/compartment_id.txt" ]; then
    COMPARTMENT_ID=$(cat /u01/aipoc/props/compartment_id.txt)
    sed -i "s|OCI_COMPARTMENT_OCID=TODO|OCI_COMPARTMENT_OCID=$COMPARTMENT_ID|g" .env
else
    echo "警告: /u01/aipoc/props/compartment_id.txt が見つかりません！"
fi

ADB_NAME=$(cat /u01/aipoc/props/adb_name.txt 2>/dev/null || true)
if [ -n "$ADB_NAME" ]; then 
    sed -i "s|ADB_NAME=TODO|ADB_NAME=$ADB_NAME|g" .env
fi

# Add ADB OCID if available
if [ -f "/u01/aipoc/props/adb_ocid.txt" ]; then
    ADB_OCID=$(cat /u01/aipoc/props/adb_ocid.txt)
    sed -i "s|ADB_OCID=ocid1.autonomousdatabase.oc1..|ADB_OCID=$ADB_OCID|g" .env
fi

# Set Oracle Client Library Directory
sed -i "s|ORACLE_CLIENT_LIB_DIR=.*|ORACLE_CLIENT_LIB_DIR=${INSTANTCLIENT_DIR}|g" .env

# Setup wallet
echo "ウォレットをセットアップ中..."
if [ -f "/u01/aipoc/props/wallet.zip" ]; then
    echo "ウォレットを展開中..."
    mkdir -p /u01/aipoc/props/wallet
    unzip -o /u01/aipoc/props/wallet.zip -d /u01/aipoc/props/wallet
    
    echo "sqlnet.oraを設定中..."
    sed -i 's|DIRECTORY="?\+/network/admin" *|DIRECTORY="/u01/aipoc/props/wallet"|g' /u01/aipoc/props/wallet/sqlnet.ora
    
    export TNS_ADMIN=/u01/aipoc/props/wallet
    echo "TNS_ADMIN=$TNS_ADMIN"
    
    # Write to profile if not already present
    if ! grep -q "TNS_ADMIN=/u01/aipoc/props/wallet" /etc/profile; then
        echo "export TNS_ADMIN=/u01/aipoc/props/wallet" >> /etc/profile
    fi
    
    echo "ウォレットファイルリスト:"
    ls -la $TNS_ADMIN
    
    echo "sqlnet.ora内容:"
    cat $TNS_ADMIN/sqlnet.ora
else
    echo "警告: /u01/aipoc/props/wallet.zip が見つかりません！"
fi

# Application setup
echo "アプリケーションを設定中..."
EXTERNAL_IP=$(curl -s -m 10 http://whatismyip.akamai.com/ || echo "")
echo "外部IP: $EXTERNAL_IP"

if [ -n "$EXTERNAL_IP" ]; then
    sed -i "s|^EXTERNAL_IP=.*|EXTERNAL_IP=$EXTERNAL_IP|g" .env
else
    echo "警告: EXTERNAL_IPの検出に失敗しました"
fi

# Accept conda pkgs
echo "Condaパッケージを承認中..."
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create conda environment
echo "Conda環境を作成中..."
if ! conda env list | grep -q "no.1-sql-assist"; then
    conda create -n no.1-sql-assist python=3.11 -y
else
    echo "Conda環境 'no.1-sql-assist' は既に存在します。"
fi

# Activate and install dependencies
echo "依存関係をインストール中..."
conda activate no.1-sql-assist
pip install -r requirements.txt

# Run application
echo "アプリケーションを起動中..."
chmod +x main.sh
nohup ./main.sh > /var/log/no1-sql-assist.log 2>&1 &

echo "初期化が完了しました。"