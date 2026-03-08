#!/bin/bash
# Launch an EC2 spot instance for auditor agent extraction.
#
# Prerequisites:
#   - AWS CLI configured (`aws configure`)
#   - 32 vCPU spot instance quota approved
#   - SSH key pair created: aws ec2 create-key-pair --key-name auditor-key \
#       --query 'KeyMaterial' --output text > ~/.ssh/auditor-key.pem && chmod 400 ~/.ssh/auditor-key.pem
#
# Required environment variables:
#   AWS_KEY_NAME     - EC2 key pair name (e.g., "auditor-key")
#
# Optional environment variables:
#   AWS_DEFAULT_REGION  - AWS region (default: us-east-2)
#   AWS_SECURITY_GROUP  - Security group ID (created automatically if empty)
#
# Usage:
#   export AWS_KEY_NAME="auditor-key"
#   bash llm_judge_feature_extraction/auditor_agent/launch_spot.sh

set -euo pipefail

INSTANCE_TYPE="m7i.4xlarge"  # 16 vCPUs, 64 GB RAM — containers are mostly idle
REGION="${AWS_DEFAULT_REGION:-us-east-2}"
KEY_NAME="${AWS_KEY_NAME:?Set AWS_KEY_NAME to your EC2 key pair name}"
SECURITY_GROUP="${AWS_SECURITY_GROUP:-}"
EBS_SIZE_GB=500
SPOT_MAX_PRICE="0.40"  # On-demand is ~$0.81

echo "=== Launching EC2 Spot Instance ==="
echo "Instance type: $INSTANCE_TYPE"
echo "Region: $REGION"
echo "Key pair: $KEY_NAME"

# Auto-detect latest Amazon Linux 2023 AMI
echo "Auto-detecting latest Amazon Linux 2023 AMI..."
AMI_ID=$(aws ec2 describe-images \
    --region "$REGION" \
    --owners amazon \
    --filters \
        "Name=name,Values=al2023-ami-2023*-x86_64" \
        "Name=state,Values=available" \
    --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
    --output text)
echo "AMI: $AMI_ID"

# Create security group if not provided
if [ -z "$SECURITY_GROUP" ]; then
    SG_NAME="auditor-agent-sg"
    echo "Creating security group: $SG_NAME"
    SECURITY_GROUP=$(aws ec2 create-security-group \
        --region "$REGION" \
        --group-name "$SG_NAME" \
        --description "Auditor agent - SSH access" \
        --query 'GroupId' --output text 2>/dev/null || \
        aws ec2 describe-security-groups \
            --region "$REGION" \
            --group-names "$SG_NAME" \
            --query 'SecurityGroups[0].GroupId' --output text)

    MY_IP="$(curl -s https://checkip.amazonaws.com)/32"
    echo "Restricting SSH to current IP: $MY_IP"
    aws ec2 authorize-security-group-ingress \
        --region "$REGION" \
        --group-id "$SECURITY_GROUP" \
        --protocol tcp --port 22 --cidr "$MY_IP" 2>/dev/null || true
    echo "Security group: $SECURITY_GROUP"
fi

S3_BUCKET="fulcrum-auditor-agent-results-2026"
IAM_ROLE_NAME="AuditorEC2Role"

# Launch spot instance
echo "Launching spot instance..."
INSTANCE_ID=$(aws ec2 run-instances \
    --region "$REGION" \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SECURITY_GROUP" \
    --iam-instance-profile "Name=$IAM_ROLE_NAME" \
    --instance-market-options '{"MarketType":"spot","SpotOptions":{"MaxPrice":"'"$SPOT_MAX_PRICE"'","SpotInstanceType":"one-time"}}' \
    --instance-initiated-shutdown-behavior terminate \
    --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":'"$EBS_SIZE_GB"',"VolumeType":"gp3","Iops":3000,"Throughput":125,"DeleteOnTermination":true}}]' \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=auditor-agent},{Key=S3Bucket,Value='"$S3_BUCKET"'}]' \
    --query 'Instances[0].InstanceId' \
    --output text)

echo "Instance ID: $INSTANCE_ID"

# Wait for instance to be running
echo "Waiting for instance to start..."
aws ec2 wait instance-running --region "$REGION" --instance-ids "$INSTANCE_ID"

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances \
    --region "$REGION" \
    --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo ""
echo "=== Instance Ready ==="
echo "Instance ID: $INSTANCE_ID"
echo "Public IP:   $PUBLIC_IP"
echo ""
echo "SSH command:"
echo "  ssh -i ~/.ssh/${KEY_NAME}.pem ec2-user@${PUBLIC_IP}"
echo ""
echo "Next steps:"
echo "  1. SSH into the instance"
echo "  2. Install git and clone the repo (private — use a GitHub PAT with repo read access):"
echo "     sudo dnf install -y git"
echo "     git clone https://<YOUR_PAT>@github.com/dariakryvosheieva/agent-psychometrics.git model_irt"
echo "  3. cd model_irt && bash llm_judge_feature_extraction/auditor_agent/setup_instance.sh"
echo "  4. Log out and back in (for Docker group), then:"
echo "     cd model_irt && source .venv/bin/activate && bash llm_judge_feature_extraction/auditor_agent/run_all_auditor.sh"
echo ""
echo "=== Auto-termination ==="
echo "The instance will automatically upload results to S3 and terminate itself"
echo "when run_all_auditor.sh completes. No manual cleanup needed."
echo ""
echo "S3 bucket: $S3_BUCKET"
echo "Download results when done:"
echo "  aws s3 sync s3://$S3_BUCKET/auditor_features/ ./auditor_features/"
echo ""
echo "To manually terminate early:"
echo "  aws ec2 terminate-instances --region $REGION --instance-ids $INSTANCE_ID"
