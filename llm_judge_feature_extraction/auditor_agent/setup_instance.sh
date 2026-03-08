#!/bin/bash
# Set up an EC2 instance for auditor agent extraction.
#
# Run this ONCE after SSH-ing into the instance.
#
# Usage:
#   ssh -i ~/.ssh/auditor-key.pem ec2-user@<IP>
#   sudo dnf install -y git
#   git clone https://github.com/<your-org>/model_irt.git
#   cd model_irt
#   bash llm_judge_feature_extraction/auditor_agent/setup_instance.sh

set -euo pipefail

echo "=== Setting up EC2 instance for auditor extraction ==="

# Install Docker + Docker Compose plugin
echo "Installing Docker..."
sudo dnf install -y docker
sudo systemctl enable docker
sudo usermod -aG docker ec2-user

# Expand Docker network address pool (default pool is too small for 30+ containers)
echo "Configuring Docker network pool..."
sudo tee /etc/docker/daemon.json > /dev/null << 'DAEMONJSON'
{
  "default-address-pools": [
    {"base": "172.17.0.0/12", "size": 24}
  ]
}
DAEMONJSON
sudo systemctl start docker

echo "Installing Docker Compose plugin..."
sudo mkdir -p /usr/local/lib/docker/cli-plugins
sudo curl -SL "https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64" \
    -o /usr/local/lib/docker/cli-plugins/docker-compose
sudo chmod +x /usr/local/lib/docker/cli-plugins/docker-compose

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
pip install -r requirements.txt

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

# Prompt for HuggingFace token (needed for gated datasets like GSO)
if [ -z "${HF_TOKEN:-}" ]; then
    echo ""
    echo "Enter your HuggingFace token (input hidden):"
    read -rs HF_KEY
    echo ""
    echo "export HF_TOKEN='${HF_KEY}'" >> ~/.bashrc
    export HF_TOKEN="${HF_KEY}"
    echo "HF token saved to ~/.bashrc"
else
    echo "HF_TOKEN already set"
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
echo "  bash llm_judge_feature_extraction/auditor_agent/run_all_auditor.sh"
