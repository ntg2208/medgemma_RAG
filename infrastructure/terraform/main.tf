terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# Data source for AWS account ID
data "aws_caller_identity" "current" {}

# S3 Bucket for Model Cache
resource "aws_s3_bucket" "models_cache" {
  bucket = var.s3_models_bucket != "" ? var.s3_models_bucket : "medgemma-models-${data.aws_caller_identity.current.account_id}"

  # Prevent accidental deletion of model cache
  lifecycle {
    prevent_destroy = true
  }

  tags = {
    Name    = "medgemma-models-cache"
    Project = "medgemma-rag"
    Purpose = "Cache for MedGemma and EmbeddingGemma models"
  }
}

# Enable versioning for safety
resource "aws_s3_bucket_versioning" "models_cache" {
  bucket = aws_s3_bucket.models_cache.id

  versioning_configuration {
    status = "Enabled"
  }
}

# Block public access
resource "aws_s3_bucket_public_access_block" "models_cache" {
  bucket = aws_s3_bucket.models_cache.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# IAM Role for EC2 to access S3
resource "aws_iam_role" "model_server" {
  name = "medgemma-model-server-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })

  tags = {
    Name    = "medgemma-model-server-role"
    Project = "medgemma-rag"
  }
}

# IAM Policy for S3 access
resource "aws_iam_role_policy" "s3_models_access" {
  name = "s3-models-access"

  role = aws_iam_role.model_server.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket",
          "s3:GetBucketLocation"
        ]
        Resource = [
          aws_s3_bucket.models_cache.arn,
          "${aws_s3_bucket.models_cache.arn}/*"
        ]
      }
    ]
  })
}

# IAM Instance Profile
resource "aws_iam_instance_profile" "model_server" {
  name = "medgemma-model-server-profile"
  role = aws_iam_role.model_server.name
}

# Security Group
resource "aws_security_group" "model_server" {
  name        = "medgemma-model-server-sg"
  description = "Security group for MedGemma model server"

  # SSH access
  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = var.allowed_cidr_blocks
  }

  # vLLM API
  ingress {
    description = "vLLM API"
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = var.allowed_cidr_blocks
  }

  # TEI Embeddings
  ingress {
    description = "TEI Embeddings"
    from_port   = 8001
    to_port     = 8001
    protocol    = "tcp"
    cidr_blocks = var.allowed_cidr_blocks
  }

  # Outbound traffic
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name    = "medgemma-model-server-sg"
    Project = "medgemma-rag"
  }
}

