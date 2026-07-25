data "oci_core_subnet" "selected_compute_subnet" {
  subnet_id = var.subnet_ai_subnet_id
}

data "template_file" "cloud_init_file" {
  template = file("./cloud_init/bootstrap.template.yaml")

  vars = {
    comp_id             = var.compartment_ocid
    db_conn             = base64gzip("admin/${var.adb_password}@${lower(var.adb_name)}_high")
    db_pass             = base64gzip(var.adb_password)
    db_dsn              = "${lower(var.adb_name)}_high"
    app_admin_password  = base64gzip(var.app_admin_password)
    vpd_login_users     = base64gzip(var.vpd_login_users)
    vpd_shared_password = base64gzip(var.vpd_shared_password)
    vpd_runtime_connection = base64gzip(
      trimspace(var.vpd_login_users) == ""
      ? ""
      : "SQL_ASSIST_RUNTIME/${var.vpd_runtime_password}@${lower(var.adb_name)}_high"
    )
    adb_name            = var.adb_name
    adb_ocid            = oci_database_autonomous_database.generated_database_autonomous_database.id
    application_port    = var.application_port
    application_git_tag = var.application_git_tag
    wallet_content      = data.external.wallet_files.result.wallet_content
  }
}


data "template_cloudinit_config" "cloud_init" {
  gzip          = true
  base64_encode = true

  part {
    filename     = "bootstrap.yaml"
    content_type = "text/cloud-config"
    content      = data.template_file.cloud_init_file.rendered
  }
}
