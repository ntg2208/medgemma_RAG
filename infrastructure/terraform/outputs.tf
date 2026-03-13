output "region" {
  description = "AWS region where infrastructure is deployed"
  value       = var.aws_region
}

output "security_group_id" {
  description = "Security group ID for model server instances"
  value       = aws_security_group.model_server.id
}

output "instance_profile_name" {
  description = "IAM instance profile name for EC2 to access S3"
  value       = aws_iam_instance_profile.model_server.name
}

output "s3_models_bucket" {
  description = "S3 bucket for model cache"
  value       = aws_s3_bucket.models_cache.id
}

output "s3_models_bucket_arn" {
  description = "S3 bucket ARN"
  value       = aws_s3_bucket.models_cache.arn
}
