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

# Spot Instance Request
resource "aws_spot_instance_request" "model_server" {
  ami                    = var.ami_id
  instance_type          = var.instance_type
  key_name               = var.key_pair_name
  vpc_security_group_ids = [aws_security_group.model_server.id]

  # Spot configuration
  spot_type                      = "persistent"
  instance_interruption_behavior = "stop"
  wait_for_fulfillment           = true

  # Root volume
  root_block_device {
    volume_size           = var.root_volume_size
    volume_type           = "gp3"
    delete_on_termination = false

    tags = {
      Name = "medgemma-model-server-root"
    }
  }

  tags = {
    Name    = "medgemma-model-server"
    Project = "medgemma-rag"
    Type    = "spot-instance"
  }
}
