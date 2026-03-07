#!/bin/bash
# Set up an EC2 instance for auditor agent extraction.
#
# Run this ONCE after SSH-ing into the instance.
#
# Usage:
#   ssh -i ~/.ssh/auditor-key.pem ec2-user@<IP>
#   git clone https://github.com/<your-org>/model_irt.git
#   cd model_irt
#   bash aws_setup/setup_instance.sh

set -euo pipefail

echo "=== Setting up EC2 instance for auditor extraction ==="

# Install Docker
echo "Installing Docker..."
sudo dnf install -y docker
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker ec2-user

# Install Python 3.12 and dev tools
echo "Installing Python..."
sudo dnf install -y python3.12 python3.12-pip python3.12-devel git

# Create virtual environment
echo "Creating Python virtual environment..."
python3.12 -m venv .venv
source .venv/bin/activate

# Install dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt 2>/dev/null || pip install \
    inspect-ai \
    inspect-evals \
    datasets \
    pandas \
    pyyaml \
    anthropic

# Clone terminal-bench-2 repo (needed for Terminal Bench task metadata)
if [ ! -d "terminal-bench-2" ]; then
    echo "Cloning terminal-bench-2 repo..."
    git clone https://github.com/harbor-framework/terminal-bench-2.git
else
    echo "terminal-bench-2 repo already exists, skipping clone"
fi

# Prompt for Anthropic API key
if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo ""
    echo "Enter your Anthropic API key (input hidden):"
    read -rs API_KEY
    echo ""
    echo "export ANTHROPIC_API_KEY='${API_KEY}'" >> ~/.bashrc
    export ANTHROPIC_API_KEY="${API_KEY}"
    echo "API key saved to ~/.bashrc"
else
    echo "ANTHROPIC_API_KEY already set"
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "IMPORTANT: Log out and back in for Docker group permissions to take effect:"
echo "  exit"
echo "  ssh -i ~/.ssh/auditor-key.pem ec2-user@<IP>"
echo ""
echo "Then run:"
echo "  cd model_irt && source .venv/bin/activate"
echo "  bash aws_setup/run_all_auditor.sh"
