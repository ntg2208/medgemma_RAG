#!/bin/bash
# Show running EC2 spot instance details
# Usage: ./scripts/status.sh

INSTANCE_NAME="medgemma-model-server"
REGION=$(terraform -chdir=infrastructure/terraform output -raw region 2>/dev/null || echo "us-east-1")

INSTANCE_JSON=$(aws ec2 describe-instances \
  --region "$REGION" \
  --filters "Name=tag:Name,Values=$INSTANCE_NAME" \
            "Name=instance-state-name,Values=running,pending,stopping,stopped" \
  --query 'Reservations[0].Instances[0].{
      InstanceId:InstanceId,
      PublicIpAddress:PublicIpAddress,
      InstanceType:InstanceType,
      State:State.Name,
      LaunchTime:LaunchTime,
      SpotRequestId:SpotInstanceRequestId
    }' \
  --output json)

INSTANCE_ID=$(echo "$INSTANCE_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('InstanceId') or '')" 2>/dev/null)

if [ -z "$INSTANCE_ID" ] || [ "$INSTANCE_ID" = "None" ]; then
  echo "No active instance found (name: $INSTANCE_NAME, region: $REGION)"
  echo "Launch with: ./scripts/start.sh"
  exit 0
fi

PUBLIC_IP=$(echo "$INSTANCE_JSON"    | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('PublicIpAddress') or 'N/A')")
INSTANCE_TYPE=$(echo "$INSTANCE_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('InstanceType') or 'N/A')")
STATE=$(echo "$INSTANCE_JSON"        | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('State') or 'N/A')")
LAUNCH_TIME=$(echo "$INSTANCE_JSON"  | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('LaunchTime') or 'N/A')")
SPOT_REQ=$(echo "$INSTANCE_JSON"     | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('SpotRequestId') or 'N/A')")

# Calculate running time
UPTIME="N/A"
if [ "$LAUNCH_TIME" != "N/A" ]; then
  LAUNCH_EPOCH=$(date -d "$LAUNCH_TIME" +%s 2>/dev/null || date -j -f "%Y-%m-%dT%H:%M:%S+00:00" "$LAUNCH_TIME" +%s 2>/dev/null || echo "")
  if [ -n "$LAUNCH_EPOCH" ]; then
    NOW_EPOCH=$(date +%s)
    ELAPSED=$(( NOW_EPOCH - LAUNCH_EPOCH ))
    HOURS=$(( ELAPSED / 3600 ))
    MINUTES=$(( (ELAPSED % 3600) / 60 ))
    UPTIME="${HOURS}h ${MINUTES}m"
  fi
fi

echo ""
echo "Instance:      $INSTANCE_ID"
echo "State:         $STATE"
echo "Type:          $INSTANCE_TYPE"
echo "Public IP:     $PUBLIC_IP"
echo "Launch time:   $LAUNCH_TIME"
echo "Uptime:        $UPTIME"
echo "Spot request:  $SPOT_REQ"
echo "Region:        $REGION"
echo ""
if [ "$PUBLIC_IP" != "N/A" ]; then
  echo "SSH:    ssh medgemma-gpu   (or: ssh -i ~/.ssh/medgemma-key.pem ubuntu@$PUBLIC_IP)"
  echo "vLLM:   http://$PUBLIC_IP:8000"
  echo "TEI:    http://$PUBLIC_IP:8001"
fi
echo ""
