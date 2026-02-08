output "instance_id" {
  description = "EC2 spot instance ID"
  value       = aws_spot_instance_request.model_server.spot_instance_id
}

output "public_ip" {
  description = "Public IP address (note: changes on stop/start)"
  value       = aws_spot_instance_request.model_server.public_ip
}

output "security_group_id" {
  description = "Security group ID"
  value       = aws_security_group.model_server.id
}

output "ssh_command" {
  description = "SSH command to connect"
  value       = "ssh -i ~/.ssh/${var.key_pair_name}.pem ubuntu@${aws_spot_instance_request.model_server.public_ip}"
}

output "vllm_endpoint" {
  description = "vLLM API endpoint"
  value       = "http://${aws_spot_instance_request.model_server.public_ip}:8000/v1"
}

output "tei_endpoint" {
  description = "TEI embeddings endpoint"
  value       = "http://${aws_spot_instance_request.model_server.public_ip}:8001"
}
