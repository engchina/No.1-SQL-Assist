
resource "oci_database_autonomous_database" "generated_database_autonomous_database" {
  admin_password                                 = var.adb_password
  autonomous_maintenance_schedule_type           = "REGULAR"
  backup_retention_period_in_days                = 1
  character_set                                  = "AL32UTF8"
  compartment_id                                 = var.compartment_ocid
  compute_count                                  = 2
  compute_model                                  = "ECPU"
  data_storage_size_in_tbs                       = 1
  db_name                                        = var.adb_name
  db_version                                     = "26ai"
  db_workload                                    = "DW"
  display_name                                   = var.adb_name
  is_auto_scaling_enabled                        = false
  is_auto_scaling_for_storage_enabled            = false
  is_dedicated                                   = false
  is_mtls_connection_required                    = true
  is_preview_version_with_service_terms_accepted = false
  license_model                                  = var.license_model
  ncharacter_set                                 = "AL16UTF16"
}

# ウォレットをダウンロード
resource "oci_database_autonomous_database_wallet" "generated_autonomous_data_warehouse_wallet" {
  autonomous_database_id = oci_database_autonomous_database.generated_database_autonomous_database.id
  password               = var.adb_password
  base64_encode_content  = "true"
  generate_type          = "SINGLE"
}

# ウォレットZIPをローカルに保存（base64デコードしてバイナリで書き込み）
resource "local_file" "wallet_zip" {
  content_base64 = oci_database_autonomous_database_wallet.generated_autonomous_data_warehouse_wallet.content
  filename       = "${path.module}/wallet_full.zip"
}

# 外部データソースでウォレットから個別ファイルを抽出（不要ファイル除外）
data "external" "wallet_files" {
  depends_on = [local_file.wallet_zip]
  program = ["bash", "-c", <<-EOT
    set -e
    WORK_DIR="${path.module}"
    cd "$WORK_DIR"
    
    # 一時ディレクトリを作成
    rm -rf wallet_extracted 2>/dev/null || true
    mkdir -p wallet_extracted
    
    # ZIPを展開
    unzip -q wallet_full.zip -d wallet_extracted 2>/dev/null
    
    # 不要ファイルを削除（README、Java関連ファイル）
    rm -f wallet_extracted/README 2>/dev/null || true
    rm -f wallet_extracted/keystore.jks 2>/dev/null || true
    rm -f wallet_extracted/truststore.jks 2>/dev/null || true
    rm -f wallet_extracted/ojdbc.properties 2>/dev/null || true
    rm -f wallet_extracted/ewallet.pem 2>/dev/null || true
    
    # 小さいZIPを作成
    cd wallet_extracted
    zip -q ../wallet_small.zip * 2>/dev/null
    cd ..
    
    # 小さいZIPをbase64エンコード
    WALLET_CONTENT=$(base64 -w 0 wallet_small.zip)
    
    # JSONとして出力
    echo "{\"wallet_content\":\"$WALLET_CONTENT\"}"
    
    # クリーンアップ
    rm -rf wallet_extracted wallet_full.zip wallet_small.zip 2>/dev/null || true
  EOT
  ]
}