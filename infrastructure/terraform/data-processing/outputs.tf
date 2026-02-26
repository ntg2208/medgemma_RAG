output "instance_id" {
  description = "EC2 spot instance ID"
  value       = aws_spot_instance_request.data_processing.spot_instance_id
}

output "region" {
  description = "AWS region where instance is deployed"
  value       = var.aws_region
}

output "public_ip" {
  description = "Public IP address (note: changes on stop/start)"
  value       = aws_spot_instance_request.data_processing.public_ip
}

output "security_group_id" {
  description = "Security group ID"
  value       = aws_security_group.data_processing.id
}

output "ssh_command" {
  description = "SSH command to connect"
  value       = "ssh -i ~/.ssh/${var.key_pair_name}.pem ubuntu@${aws_spot_instance_request.data_processing.public_ip}"
}

output "jupyter_url" {
  description = "Jupyter Notebook URL"
  value       = "http://${aws_spot_instance_request.data_processing.public_ip}:8888"
}

output "gradio_url" {
  description = "Gradio UI URL"
  value       = "http://${aws_spot_instance_request.data_processing.public_ip}:7860"
}
