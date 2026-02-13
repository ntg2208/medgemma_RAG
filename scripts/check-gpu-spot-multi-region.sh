#!/bin/bash
#
# Multi-region GPU spot availability checker
# Checks g5.xlarge and g6.xlarge across multiple US regions
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Instance types to check
INSTANCE_TYPES=("g5.xlarge" "g6.xlarge")

# US regions with GPU availability
REGIONS=(
    "us-east-1"  # N. Virginia
    "us-east-2"  # Ohio
    "us-west-2"  # Oregon
    "us-west-1"  # California
)

echo "========================================================================================================"
echo "Multi-Region GPU Spot Availability Checker"
echo "========================================================================================================"
echo "Checking spot pricing and availability for GPU instances..."
echo ""

# Cross-platform date calculation
if [[ "$(uname)" == "Darwin" ]]; then
  START_TIME=$(date -u -v-1H +'%Y-%m-%dT%H:%M:%S')
else
  START_TIME=$(date -u -d '1 hour ago' +'%Y-%m-%dT%H:%M:%S')
fi

# Store results for final summary (bash 3.2 compatible - no associative arrays)
KEYS=()
PRICES=()
ZONES=()

# Function to check a single region/instance combination
check_instance() {
    local region=$1
    local instance_type=$2

    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}Region: $region | Instance: $instance_type${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    # Get spot price history
    local prices=$(aws ec2 describe-spot-price-history \
        --instance-types "$instance_type" \
        --region "$region" \
        --start-time "$START_TIME" \
        --product-descriptions "Linux/UNIX" \
        --query 'SpotPriceHistory[*].[AvailabilityZone,SpotPrice,Timestamp]' \
        --output text 2>/dev/null)

    if [ -z "$prices" ]; then
        echo -e "${RED}✗ No spot pricing data available${NC}"
        echo ""
        return 1
    fi

    # Parse and display prices by AZ
    local min_price=9999
    local best_az=""

    while IFS=$'\t' read -r az price timestamp; do
        if [ ! -z "$az" ]; then
            printf "  %-15s: \$%-8s (as of %s)\n" "$az" "$price" "$timestamp"

            # Track minimum price
            if (( $(echo "$price < $min_price" | bc -l) )); then
                min_price=$price
                best_az=$az
            fi
        fi
    done <<< "$prices"

    # Store best price for this combination (bash 3.2 compatible)
    local key="${region}:${instance_type}"
    KEYS+=("$key")
    PRICES+=("$min_price")
    ZONES+=("$best_az")

    # Color-code the result
    if (( $(echo "$min_price < 0.35" | bc -l) )); then
        echo -e "${GREEN}✓ Best price: \$$min_price/hr in $best_az (GOOD)${NC}"
    elif (( $(echo "$min_price < 0.50" | bc -l) )); then
        echo -e "${YELLOW}⚠ Best price: \$$min_price/hr in $best_az (MODERATE)${NC}"
    else
        echo -e "${RED}✗ Best price: \$$min_price/hr in $best_az (HIGH)${NC}"
    fi

    echo ""
    return 0
}

# Check all combinations
echo "Fetching spot prices across regions (this may take 30-60 seconds)..."
echo ""

for region in "${REGIONS[@]}"; do
    for instance_type in "${INSTANCE_TYPES[@]}"; do
        check_instance "$region" "$instance_type"
    done
done

# Print summary recommendation
echo "========================================================================================================"
echo "RECOMMENDATION SUMMARY"
echo "========================================================================================================"
echo ""

# Find the absolute best option
best_overall_price=9999
best_overall_key=""
best_overall_index=-1

# Find the absolute best option by iterating over all stored results
for i in "${!KEYS[@]}"; do
    price=${PRICES[$i]}
    if (( $(echo "$price < $best_overall_price" | bc -l) )); then
        best_overall_price=$price
        best_overall_key=${KEYS[$i]}
        best_overall_index=$i
    fi
done

if [ ! -z "$best_overall_key" ]; then
    IFS=':' read -r best_region best_instance <<< "$best_overall_key"
    best_zone=${ZONES[$best_overall_index]}

    echo -e "${GREEN}🏆 BEST OPTION${NC}"
    echo "   Region:        $best_region"
    echo "   Instance:      $best_instance"
    echo "   Price:         \$$best_overall_price/hr"
    echo "   Zone:          $best_zone"
    echo ""

    # Show top 3 alternatives
    echo -e "${YELLOW}📊 TOP 3 ALTERNATIVES${NC}"

    # Sort all options by price (bash 3.2 compatible)
    # Create temporary file with price:key:zone entries
    TMPFILE=$(mktemp /tmp/spot_prices.XXXXXX)
    for i in "${!KEYS[@]}"; do
        echo "${PRICES[$i]}:${KEYS[$i]}:${ZONES[$i]}" >> "$TMPFILE"
    done

    # Sort by price (numeric) and get top 3
    sorted_entries=$(sort -n "$TMPFILE" | head -n 3)
    rm "$TMPFILE"

    # Parse sorted entries
    local rank=1
    while IFS=':' read -r price key zone; do
        if [ "$key" != "$best_overall_key" ]; then
            IFS=':' read -r region instance <<< "$key"
            echo "   $rank. $region | $instance | \$$price/hr ($zone)"
            ((rank++))
        fi
    done <<< "$sorted_entries"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "NEXT STEPS:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "1. Update your terraform.tfvars:"
    echo "   aws_region    = \"$best_region\""
    echo "   instance_type = \"$best_instance\""
    echo ""
    echo "2. Verify AMI ID for the region (if different from current):"
    echo "   aws ec2 describe-images \\"
    echo "     --region $best_region \\"
    echo "     --owners amazon \\"
    echo "     --filters 'Name=name,Values=Deep Learning OSS Nvidia Driver AMI GPU PyTorch*' \\"
    echo "     --query 'Images | sort_by(@, &CreationDate)[-1].[ImageId,Name]' \\"
    echo "     --output table"
    echo ""
    echo "3. Deploy with Terraform:"
    echo "   cd infrastructure/terraform"
    echo "   terraform apply"
    echo ""
else
    echo -e "${RED}✗ No spot pricing data available in any checked region${NC}"
    echo "This may indicate:"
    echo "  - AWS CLI not configured correctly"
    echo "  - Network connectivity issues"
    echo "  - No spot capacity currently available (unlikely)"
fi

echo "========================================================================================================"
echo "NOTE: Prices fluctuate. Run this script before each deployment to get current rates."
echo "========================================================================================================"
