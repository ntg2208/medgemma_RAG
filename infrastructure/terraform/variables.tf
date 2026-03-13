variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-2"
}

variable "instance_type" {
  description = "EC2 instance type (g5.xlarge or g6.xlarge recommended for A10G/L4 GPU)"
  type        = string
  default     = "g6.xlarge"

  validation {
    condition     = contains(["g5.xlarge", "g6.xlarge", "g5.2xlarge", "g6.2xlarge"], var.instance_type)
    error_message = "Instance type must support bfloat16 (compute capability ≥8.0). Recommended: g5.xlarge or g6.xlarge"
  }
}

variable "ami_id" {
  description = "AMI ID for Deep Learning AMI"
  type        = string
  default     = "ami-01bc785757b863550" # Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.6.0 (Ubuntu 22.04) us-east-2
}

variable "key_pair_name" {
  description = "Name of the EC2 key pair"
  type        = string
  default     = "medgemma-key"
}

variable "allowed_cidr_blocks" {
  description = "CIDR blocks allowed for SSH and API access"
  type        = list(string)
  default     = ["31.221.62.58/32"]
}

variable "s3_models_bucket" {
  description = "S3 bucket name for cached models (leave empty to create new bucket)"
  type        = string
  default     = "" # Will auto-generate if empty
}
