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

# Security Group for Data Processing Instance
resource "aws_security_group" "data_processing" {
  name        = "medgemma-data-processing-sg"
  description = "Security group for MedGemma data processing/coding instance"

  # SSH access
  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = var.allowed_cidr_blocks
  }

  # Jupyter Notebook
  ingress {
    description = "Jupyter Notebook"
    from_port   = 8888
    to_port     = 8888
    protocol    = "tcp"
    cidr_blocks = var.allowed_cidr_blocks
  }

  # Gradio UI
  ingress {
    description = "Gradio UI"
    from_port   = 7860
    to_port     = 7860
    protocol    = "tcp"
    cidr_blocks = var.allowed_cidr_blocks
  }

  # Docling API (if running as server)
  ingress {
    description = "Docling API"
    from_port   = 5000
    to_port     = 5000
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
    Name    = "medgemma-data-processing-sg"
    Project = "medgemma-rag"
  }
}

# Spot Instance Request for Data Processing
resource "aws_spot_instance_request" "data_processing" {
  ami                    = var.ami_id
  instance_type          = var.instance_type
  key_name               = var.key_pair_name
  vpc_security_group_ids = [aws_security_group.data_processing.id]

  # Spot configuration
  spot_type                      = "one-time"
  instance_interruption_behavior = "terminate"
  wait_for_fulfillment           = true

  # Root volume
  root_block_device {
    volume_size           = var.root_volume_size
    volume_type           = "gp3"
    delete_on_termination = false

    tags = {
      Name = "medgemma-data-processing-root"
    }
  }

  tags = {
    Name    = "medgemma-data-processing"
    Project = "medgemma-rag"
    Type    = "spot-instance"
    Purpose = "coding-and-ocr"
  }
}
