terraform {
  required_providers {
    external = {
      source  = "hashicorp/external"
      version = ">= 2.0.0"
    }
    local = {
      source  = "hashicorp/local"
      version = ">= 2.0.0"
    }
    oci = {
      source  = "oracle/oci"
      version = ">= 4.67.3"
    }
    template = {
      source  = "hashicorp/template"
      version = ">= 2.2.0"
    }
  }
}
