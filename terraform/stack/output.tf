locals {
  instance_access_ip = local.compute_subnet_prohibits_public_ip ? oci_core_instance.generated_oci_core_instance.private_ip : oci_core_instance.generated_oci_core_instance.public_ip
}

output "autonomous_data_warehouse_admin_password" {
  value     = var.adb_password
  sensitive = true
}

output "autonomous_data_warehouse_ocid" {
  description = "Autonomous Database OCID"
  value       = oci_database_autonomous_database.generated_database_autonomous_database.id
}

output "autonomous_data_warehouse_high_connection_string" {
  value = lookup(
    oci_database_autonomous_database.generated_database_autonomous_database.connection_strings[0].all_connection_strings,
    "HIGH",
    "unavailable",
  )
}

output "ssh_to_instance" {
  description = "convenient command to ssh to the instance using its public or private access IP"
  value       = "ssh -o ServerAliveInterval=10 ubuntu@${local.instance_access_ip}"
}

output "application_url" {
  description = "convenient URL to access the application using its public or private access IP"
  value       = "http://${local.instance_access_ip}"
}
