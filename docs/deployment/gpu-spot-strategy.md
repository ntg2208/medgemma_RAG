# GPU Spot Instance Strategy

## Overview

This guide explains the multi-region, multi-instance type strategy for maximizing GPU spot availability while minimizing costs.

## Instance Options

| Instance | GPU | VRAM | Typical Spot Price | Availability | Notes |
|----------|-----|------|-------------------|--------------|-------|
| **g5.xlarge** | NVIDIA A10G | 24GB | $0.30-0.40/hr | Medium | Mature, widely used |
| **g6.xlarge** | NVIDIA L4 | 24GB | $0.25-0.35/hr | Medium-High | Newer, often better availability |
| g5.2xlarge | NVIDIA A10G | 24GB | $0.50-0.60/hr | Medium | Same GPU, 2x CPU/RAM |
| g6.2xlarge | NVIDIA L4 | 24GB | $0.45-0.55/hr | Medium | Same GPU, 2x CPU/RAM |

**Both g5 and g6 support bfloat16 (compute capability ≥8.0) required for MedGemma 1.5.**

## Region Recommendations

| Region | Code | g5.xlarge | g6.xlarge | Notes |
|--------|------|-----------|-----------|-------|
| **N. Virginia** | us-east-1 | Low | Medium | High demand, avoid if possible |
| **Ohio** | us-east-2 | Medium | Medium-High | Good balance |
| **Oregon** | us-west-2 | Medium | Medium-High | Current default, reliable |
| California | us-west-1 | Low | Low | Limited GPU capacity |

## Spot Configuration Changes

### 1. One-Time vs Persistent

**Changed from `persistent` to `one-time` (infrastructure/terraform/main.tf:70)**

| Type | Behavior | Pros | Cons |
|------|----------|------|------|
| **one-time** | Request fulfills once, done | Safe, no surprise costs | Won't auto-restart after interruption |
| persistent | Keeps trying to fulfill | Auto-recovers from interruptions | Can cause surprise costs if forgotten |

**Safety incident:** Persistent request ran 13+ hours unnoticed in us-east-1 after switching to us-east-2, costing $3 extra.

**Recommendation:** Use one-time unless you need long-running sessions (>4 hours) and can monitor actively.

### 2. Interruption Behavior

**Setting:** `instance_interruption_behavior = "stop"`

When spot capacity is reclaimed:
- Instance stops (doesn't terminate)
- EBS volumes persist (models remain in `~/models_cache/`)
- Can manually restart later
- One-time request won't auto-restart (need to create new request)

## Workflow

### Step 1: Check Availability

**Quick check (single region):**
```bash
./scripts/check-gpu-spot-availability.sh us-west-2 g5.xlarge
./scripts/check-gpu-spot-availability.sh us-west-2 g6.xlarge
```

**Comprehensive check (all regions):**
```bash
./scripts/check-gpu-spot-multi-region.sh
```

This will:
- Check g5.xlarge and g6.xlarge in all US regions
- Show spot prices by availability zone
- Recommend the best option
- Provide ready-to-use terraform.tfvars values

### Step 2: Update Configuration

Edit `infrastructure/terraform/terraform.tfvars`:

```hcl
# Use values from check-gpu-spot-multi-region.sh output
aws_region    = "us-west-2"
instance_type = "g6.xlarge"  # or g5.xlarge

# Update AMI if switching regions (see AMI IDs below)
ami_id = "ami-xxxxxxxxx"
```

### Step 3: Deploy

```bash
cd infrastructure/terraform
terraform apply
```

If spot request fails (capacity unavailable):
1. Run availability check again
2. Try alternative instance type (g5 ↔ g6)
3. Try alternative region
4. Deploy again

## AMI IDs by Region

Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.6.0 (Ubuntu 22.04):

| Region | AMI ID |
|--------|--------|
| us-east-1 | ami-0c702567ccf8b120a |
| us-east-2 | ami-01bc785757b863550 |
| us-west-2 | ami-0b1c3e116f347cc8a |
| us-west-1 | ami-0a8c9f8b4c8e5f6d7 |

**To find current AMIs:**
```bash
aws ec2 describe-images \
  --region us-west-2 \
  --owners amazon \
  --filters 'Name=name,Values=Deep Learning OSS Nvidia Driver AMI GPU PyTorch*' \
  --query 'Images | sort_by(@, &CreationDate)[-1].[ImageId,Name]' \
  --output table
```

## Cost Analysis

### Typical Monthly Costs (30 hours/month usage)

| Instance | Spot Price | Monthly Cost | Use Case |
|----------|-----------|--------------|----------|
| g6.xlarge | $0.30/hr | $9 | **Best value** |
| g5.xlarge | $0.35/hr | $10.50 | Default choice |
| g6.xlarge | $0.40/hr | $12 | High demand |
| g5.xlarge | $0.45/hr | $13.50 | Peak pricing |

**EBS storage:** ~$6/month (75GB gp3 @ $0.08/GB/month)

**Total:** $15-20/month with moderate usage

### Cost Optimization Tips

1. **Stop when not in use**: Models are cached on EBS, restart is fast (~2-3 min)
2. **Use g6 when available**: Usually 10-15% cheaper than g5
3. **Check prices before deploying**: Use `check-gpu-spot-multi-region.sh`
4. **Set billing alerts**: Get notified if costs exceed threshold
5. **Use one-time spots**: Prevents runaway costs from forgotten instances

## Cleanup Checklist

When done with spot instance (to avoid surprise costs):

```bash
# 1. Cancel spot request (if persistent - not needed for one-time)
aws ec2 describe-spot-instance-requests --region <region> \
  --query 'SpotInstanceRequests[*].[SpotInstanceRequestId,State]' --output table

aws ec2 cancel-spot-instance-requests --region <region> \
  --spot-instance-request-ids <request-id>

# 2. Terminate instance
aws ec2 terminate-instances --region <region> --instance-ids <instance-id>

# 3. Delete EBS volumes (if not needed anymore)
aws ec2 describe-volumes --region <region> \
  --filters "Name=tag:Name,Values=medgemma-model-server-root" \
  --query 'Volumes[*].[VolumeId,State]' --output table

aws ec2 delete-volume --region <region> --volume-id <volume-id>
```

**Note:** With one-time spots, you only need to terminate the instance (step 2).

## Troubleshooting

### "InsufficientInstanceCapacity"

**Cause:** No spot capacity available in that AZ

**Solutions:**
1. Run `./scripts/check-gpu-spot-multi-region.sh`
2. Try different instance type (g5 ↔ g6)
3. Try different region (often us-east-2 or us-west-2 work)
4. Wait 30-60 minutes and retry (capacity fluctuates)

### "MaxSpotInstanceCountExceeded"

**Cause:** You have too many active spot requests

**Solution:**
```bash
# List all active spot requests across all regions
for region in us-east-1 us-east-2 us-west-2; do
  echo "Region: $region"
  aws ec2 describe-spot-instance-requests --region $region \
    --filters "Name=state,Values=active,open" \
    --query 'SpotInstanceRequests[*].[SpotInstanceRequestId,InstanceType,State]' \
    --output table
done

# Cancel unneeded requests
aws ec2 cancel-spot-instance-requests --region <region> \
  --spot-instance-request-ids <request-id>
```

### Spot instance stopped unexpectedly

**Cause:** AWS reclaimed capacity (2-minute warning)

**With one-time spots:**
- Instance stops, EBS persists
- Need to create new spot request to restart
- Models remain on disk, no re-download needed

**Solution:**
```bash
# Check if your models/data persist
aws ec2 describe-volumes --region <region> \
  --filters "Name=tag:Name,Values=medgemma-model-server-root" \
  --query 'Volumes[*].[VolumeId,State,Size]'

# Create new spot request
cd infrastructure/terraform
terraform apply
```

## References

- [AWS Spot Instance Pricing](https://aws.amazon.com/ec2/spot/pricing/)
- [EC2 Instance Types - GPU](https://aws.amazon.com/ec2/instance-types/#Accelerated_Computing)
- [Spot Instance Interruptions](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/spot-interruptions.html)
