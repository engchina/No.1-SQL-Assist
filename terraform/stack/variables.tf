variable "availability_domain" {
  default = "bxtG:AP-TOKYO-1-AD-1"
}

variable "compartment_ocid" {
  default = ""
}

variable "adb_name" {
  default = "AISQLADB"
}

variable "adb_password" {
  default = ""
}

variable "adb_use_private_subnet" {
  description = "Whether to use a private subnet for Autonomous Database"
  type        = bool
  default     = false
}

variable "adb_subnet_id" {
  description = "Private subnet OCID for Autonomous Database (used when private subnet is enabled)"
  type        = string
  default     = ""
}

variable "license_model" {
  default = ""
}

variable "instance_display_name" {
  default = "AISQL_INSTANCE"
}

variable "instance_shape" {
  default = "VM.Standard.E4.Flex"
}

variable "instance_flex_shape_ocpus" {
  description = "Number of OCPUs for the Compute flex shape"
  type        = number
  default     = 2
}

variable "instance_flex_shape_memory" {
  description = "Memory in GB for the Compute flex shape"
  type        = number
  default     = 16
}

variable "instance_boot_volume_size" {
  description = "Boot block volume size in GB for the Compute instance"
  type        = number
  default     = 100
}

variable "instance_boot_volume_vpus" {
  description = "Boot block volume performance in VPUs per GB for the Compute instance"
  type        = number
  default     = 10
}

variable "instance_image_source_id" {
  default = "ocid1.image.oc1.ap-tokyo-1.aaaaaaaaoiusqhftxmiyjlulnxx5mdnqfv6pjx4hdcoks3exn7gsrcwpkpdq"
}

variable "subnet_ai_subnet_id" {
  default = ""
}

variable "ssh_authorized_keys" {
  default = ""
}
