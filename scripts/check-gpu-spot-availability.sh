#!/bin/bash
#
# Check spot availability and pricing for GPU instances
# Usage: ./check-gpu-spot-availability.sh [region] [instance-type]
# Example: ./check-gpu-spot-availability.sh us-west-2 g6.xlarge
#

set -e

REGION=${1:-"us-east-2"}
INSTANCE_TYPE=${2:-"g5.xlarge"}

# Validate instance type
if [[ ! "$INSTANCE_TYPE" =~ ^g[56]\.(xlarge|2xlarge)$ ]]; then
    echo "Warning: $INSTANCE_TYPE may not support bfloat16 required for MedGemma 1.5"
    echo "Recommended: g5.xlarge (A10G) or g6.xlarge (L4)"
fi

echo "================================================"
echo "GPU Spot Availability Check"
echo "================================================"
echo "Region: $REGION"
echo "Instance Type: $INSTANCE_TYPE"
echo ""

echo "1. Checking spot pricing history (last hour)..."
# Cross-platform date calculation (macOS uses -v flag, Linux uses -d)
if [[ "$(uname)" == "Darwin" ]]; then
  # macOS
  START_TIME=$(date -u -v-1H +'%Y-%m-%dT%H:%M:%S')
else
  # Linux
  START_TIME=$(date -u -d '1 hour ago' +'%Y-%m-%dT%H:%M:%S')
fi

aws ec2 describe-spot-price-history \
  --instance-types "$INSTANCE_TYPE" \
  --region "$REGION" \
  --start-time "$START_TIME" \
  --product-descriptions "Linux/UNIX" \
  --query 'SpotPriceHistory[*].[AvailabilityZone,SpotPrice,Timestamp]' \
  --output table

echo ""
echo "2. Checking current spot instance limits..."
aws ec2 describe-spot-instance-requests \
  --region "$REGION" \
  --filters "Name=state,Values=active,open" \
  --query 'SpotInstanceRequests[*].[InstanceType,State]' \
  --output table

echo ""
echo "3. Checking instance limits (vCPUs)..."
aws ec2 describe-account-attributes \
  --region "$REGION" \
  --attribute-names "max-instances" \
  --query 'AccountAttributes[*].[AttributeName,AttributeValues[*].AttributeValue]' \
  --output table

echo ""
echo "================================================"
echo "Interpretation:"
echo "- g5.xlarge: \$0.30-0.40/hr (A10G, 24GB)"
echo "- g6.xlarge: \$0.25-0.35/hr (L4, 24GB) - often better availability"
echo "- If capacity is unavailable, try:"
echo "  1. Different instance type (g5 ↔ g6)"
echo "  2. Different region (us-east-1, us-east-2, us-west-2)"
echo "  3. Multi-region check: ./check-gpu-spot-multi-region.sh"
echo "================================================"