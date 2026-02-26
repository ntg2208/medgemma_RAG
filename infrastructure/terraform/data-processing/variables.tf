variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "instance_type" {
  description = "EC2 instance type (g4dn.xlarge for T4 GPU - good for Docling OCR)"
  type        = string
  default     = "g4dn.xlarge"

  validation {
    condition     = contains(["g4dn.xlarge", "g4dn.2xlarge", "g4dn.4xlarge"], var.instance_type)
    error_message = "Instance type must be g4dn family (T4 GPU). Recommended: g4dn.xlarge"
  }
}

variable "ami_id" {
  description = "AMI ID for Deep Learning AMI"
  type        = string
  default     = "ami-0c702567ccf8b120a" # Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.6.0 (Ubuntu 22.04)
}

variable "key_pair_name" {
  description = "Name of the EC2 key pair"
  type        = string
  default     = "medgemma-key"
}

variable "root_volume_size" {
  description = "Root EBS volume size in GB"
  type        = number
  default     = 75
}

variable "allowed_cidr_blocks" {
  description = "CIDR blocks allowed for SSH and API access"
  type        = list(string)
  default     = ["0.0.0.0/0"] # Override with your IP for security
}
