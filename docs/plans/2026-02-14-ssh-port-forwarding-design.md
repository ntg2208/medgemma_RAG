# SSH Port Forwarding Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add SSH port forwarding for Jupyter Notebook (port 8888) and Gradio (port 7860) to enable local browser access to services running on the remote EC2 GPU instance.

**Architecture:** Modify existing `~/.ssh/config` to add `LocalForward` directives to the `medgemma-gpu` host configuration, forwarding remote ports to localhost for seamless access.

**Tech Stack:** SSH port forwarding, EC2 spot instances, Jupyter Notebook, Gradio

---

## Task 1: Check Current SSH Configuration

**Files:**
- Read: `/Users/ntg/.ssh/config`

**Step 1: Read current SSH configuration**

Run: `cat /Users/ntg/.ssh/config`
Expected: Shows current SSH config with `medgemma-gpu` host entry

**Step 2: Verify current configuration format**

Run: `grep -A 5 "Host medgemma-gpu" /Users/ntg/.ssh/config`
Expected: Shows the medgemma-gpu host configuration block

**Step 3: Check for existing port forwarding**

Run: `grep -i "localforward" /Users/ntg/.ssh/config`
Expected: May show existing port forwarding entries or none

**Step 4: Commit current state**

```bash
git add docs/plans/2026-02-14-ssh-port-forwarding-design.md
git commit -m "docs: start SSH port forwarding implementation plan"
```

---
## Task 2: Check Local Port Availability

**Files:**
- No file changes, system check only

**Step 1: Check if port 8888 is in use locally**

Run: `lsof -i :8888 2>/dev/null || echo "Port 8888 not in use"`
Expected: "Port 8888 not in use" or shows process using port

**Step 2: Check if port 7860 is in use locally**

Run: `lsof -i :7860 2>/dev/null || echo "Port 7860 not in use"`
Expected: "Port 7860 not in use" or shows process using port

**Step 3: Verify Gradio port in app.py**

Run: `grep -n "server_port" /Users/ntg/Documents/Personal_Projects/medgemma_RAG/app.py`
Expected: Shows `server_port=7860` (line 594)

**Step 4: Record port availability**

Run: `echo "Ports 8888 and 7860 available for forwarding" >> /tmp/port-check.txt && cat /tmp/port-check.txt`
Expected: Confirms ports are available

---
## Task 3: Backup Current SSH Configuration

**Files:**
- Create: `/Users/ntg/.ssh/config.backup.$(date +%Y%m%d)`

**Step 1: Create timestamped backup**

Run: `cp /Users/ntg/.ssh/config /Users/ntg/.ssh/config.backup.$(date +%Y%m%d_%H%M%S)`
Expected: Creates backup file with timestamp

**Step 2: Verify backup creation**

Run: `ls -la /Users/ntg/.ssh/config.backup.* | tail -1`
Expected: Shows backup file with correct permissions and size

**Step 3: Test backup readability**

Run: `head -20 /Users/ntg/.ssh/config.backup.* | tail -1`
Expected: Shows first line of backup file

**Step 4: Commit backup record**

```bash
echo "SSH config backup created: $(ls /Users/ntg/.ssh/config.backup.* | tail -1)" >> docs/plans/2026-02-14-ssh-port-forwarding-design.md
git add docs/plans/2026-02-14-ssh-port-forwarding-design.md
git commit -m "chore: backup SSH configuration before modification"
```

---
## Task 4: Add Port Forwarding to SSH Configuration

**Files:**
- Modify: `/Users/ntg/.ssh/config`

**Step 1: Read current medgemma-gpu configuration**

Run: `sed -n '/^Host medgemma-gpu/,/^Host/p' /Users/ntg/.ssh/config`
Expected: Shows current medgemma-gpu host block

**Step 2: Add LocalForward directives**

Edit `/Users/ntg/.ssh/config` to add after `ServerAliveCountMax 3`:
```bash
sed -i '' '/ServerAliveCountMax 3/a\
    LocalForward 8888 localhost:8888    # Jupyter Notebook\
    LocalForward 7860 localhost:7860    # Gradio UI' /Users/ntg/.ssh/config
```

**Step 3: Verify the edit**

Run: `sed -n '/^Host medgemma-gpu/,/^Host/p' /Users/ntg/.ssh/config`
Expected: Shows updated configuration with both LocalForward lines

**Step 4: Check syntax validity**

Run: `ssh -G medgemma-gpu 2>&1 | head -5`
Expected: Shows SSH configuration without errors

---
## Task 5: Test SSH Connection with Port Forwarding

**Files:**
- No file changes, connection testing only

**Step 1: Test SSH connection (dry run)**

Run: `ssh -v medgemma-gpu echo "SSH test successful" 2>&1 | grep -E "debug1:|successful" | tail -5`
Expected: Shows debug output ending with "SSH test successful"

**Step 2: Test port forwarding establishment**

Run: `ssh -N -f -L 8888:localhost:8888 -L 7860:localhost:7860 medgemma-gpu && sleep 2 && ps aux | grep "ssh.*medgemma-gpu" | grep -v grep`
Expected: Shows SSH process running in background with port forwarding

**Step 3: Check local port listening**

Run: `netstat -an | grep -E "8888|7860" | grep LISTEN`
Expected: Shows ports 8888 and 7860 listening locally

**Step 4: Clean up test connection**

Run: `pkill -f "ssh.*medgemma-gpu" && echo "Test connection cleaned up"`
Expected: "Test connection cleaned up"

---
## Task 6: Create Usage Documentation

**Files:**
- Create: `/Users/ntg/Documents/Personal_Projects/medgemma_RAG/docs/ssh-port-forwarding-guide.md`

**Step 1: Create usage guide**

```bash
cat > /Users/ntg/Documents/Personal_Projects/medgemma_RAG/docs/ssh-port-forwarding-guide.md << 'EOF'
# SSH Port Forwarding Guide

## Overview
SSH port forwarding enables local browser access to services running on the remote EC2 GPU instance:
- Jupyter Notebook: `http://localhost:8888`
- Gradio UI: `http://localhost:7860`

## Configuration
Added to `~/.ssh/config`:
```ssh
Host medgemma-gpu
    HostName [EC2_PUBLIC_IP]
    User ubuntu
    IdentityFile ~/.ssh/medgemma-key.pem
    ServerAliveInterval 60
    ServerAliveCountMax 3
    LocalForward 8888 localhost:8888    # Jupyter Notebook
    LocalForward 7860 localhost:7860    # Gradio UI
```

## Usage

### Standard Connection (Shell + Ports)
```bash
ssh medgemma-gpu
# Now access:
# - Jupyter: http://localhost:8888
# - Gradio: http://localhost:7860
```

### Port-Only Connection (Background)
```bash
ssh -N medgemma-gpu
# Establishes port forwarding without interactive shell
```

### With EC2 Workflow
```bash
# 1. Start EC2 instance
./scripts/ec2-start.sh

# 2. Get new IP (if changed)
./scripts/ec2-status.sh

# 3. Update SSH config with new IP
# Edit ~/.ssh/config: update HostName

# 4. Connect with port forwarding
ssh medgemma-gpu

# 5. Start services on EC2
cd ~/medgemma_RAG
./scripts/start-model-server.sh  # vLLM + TEI
jupyter notebook --no-browser --port=8888
python app.py  # Gradio
```

## Testing

### Verify Port Forwarding
```bash
# Check SSH process
ps aux | grep "ssh.*medgemma-gpu"

# Check local ports
netstat -an | grep -E "8888|7860" | grep LISTEN

# Test service accessibility
curl -s http://localhost:8888 | grep -i jupyter
curl -s http://localhost:7860 | grep -i gradio
```

### Troubleshooting

#### Port Conflicts
If ports 8888 or 7860 are already in use locally:
```bash
# Find conflicting process
lsof -i :8888
lsof -i :7860

# Stop conflicting process or use alternative ports
```

#### SSH Connection Issues
```bash
# Test basic SSH
ssh -v medgemma-gpu echo "test"

# Check security group rules
# Ports needed: 22 (SSH), 8000 (vLLM), 8001 (TEI), 8888 (Jupyter), 7860 (Gradio)
```

#### EC2 IP Changed
Update `HostName` in `~/.ssh/config`:
```bash
# Get new IP
./scripts/ec2-status.sh

# Edit config
nano ~/.ssh/config
# Update: HostName [NEW_IP]
```

## Notes
- Model APIs remain accessible via EC2 public IP: `http://[EC2_IP]:8000` and `http://[EC2_IP]:8001`
- Jupyter and Gradio require services to be running on EC2
- Port forwarding works alongside regular SSH shell access
EOF
```

**Step 2: Verify guide creation**

Run: `head -20 /Users/ntg/Documents/Personal_Projects/medgemma_RAG/docs/ssh-port-forwarding-guide.md`
Expected: Shows guide header and overview

**Step 3: Add guide to git**

```bash
git add /Users/ntg/Documents/Personal_Projects/medgemma_RAG/docs/ssh-port-forwarding-guide.md
git commit -m "docs: add SSH port forwarding usage guide"
```

---
## Task 7: Create Quick Test Script

**Files:**
- Create: `/Users/ntg/Documents/Personal_Projects/medgemma_RAG/scripts/test-port-forwarding.sh`

**Step 1: Create test script**

```bash
cat > /Users/ntg/Documents/Personal_Projects/medgemma_RAG/scripts/test-port-forwarding.sh << 'EOF'
#!/bin/bash

# Test SSH Port Forwarding for MedGemma RAG
# Usage: ./test-port-forwarding.sh

set -e

echo "═══════════════════════════════════════"
echo "SSH Port Forwarding Test"
echo "═══════════════════════════════════════"

# Check SSH config
echo "1. Checking SSH configuration..."
if grep -q "LocalForward 8888" ~/.ssh/config; then
    echo "   ✓ Jupyter port forwarding configured"
else
    echo "   ✗ Jupyter port forwarding NOT configured"
fi

if grep -q "LocalForward 7860" ~/.ssh/config; then
    echo "   ✓ Gradio port forwarding configured"
else
    echo "   ✗ Gradio port forwarding NOT configured"
fi

# Check local port availability
echo ""
echo "2. Checking local port availability..."
if lsof -i :8888 >/dev/null 2>&1; then
    echo "   ✗ Port 8888 is in use locally"
    lsof -i :8888 | head -3
else
    echo "   ✓ Port 8888 available locally"
fi

if lsof -i :7860 >/dev/null 2>&1; then
    echo "   ✗ Port 7860 is in use locally"
    lsof -i :7860 | head -3
else
    echo "   ✓ Port 7860 available locally"
fi

# Test SSH connection
echo ""
echo "3. Testing SSH connection..."
if ssh -q -o BatchMode=yes -o ConnectTimeout=5 medgemma-gpu exit; then
    echo "   ✓ SSH connection successful"
else
    echo "   ✗ SSH connection failed"
    echo "   Check: ~/.ssh/config, SSH key, EC2 instance status"
fi

echo ""
echo "═══════════════════════════════════════"
echo "Test Complete"
echo ""
echo "Next steps:"
echo "1. Start EC2: ./scripts/ec2-start.sh"
echo "2. Get IP: ./scripts/ec2-status.sh"
echo "3. Update ~/.ssh/config if IP changed"
echo "4. Connect: ssh medgemma-gpu"
echo "5. Start services on EC2"
echo "6. Access: http://localhost:8888 (Jupyter)"
echo "7. Access: http://localhost:7860 (Gradio)"
echo "═══════════════════════════════════════"
EOF
```

**Step 2: Make script executable**

Run: `chmod +x /Users/ntg/Documents/Personal_Projects/medgemma_RAG/scripts/test-port-forwarding.sh`
Expected: Script is now executable

**Step 3: Test script execution**

Run: `/Users/ntg/Documents/Personal_Projects/medgemma_RAG/scripts/test-port-forwarding.sh`
Expected: Shows test results without errors

**Step 4: Commit test script**

```bash
git add /Users/ntg/Documents/Personal_Projects/medgemma_RAG/scripts/test-port-forwarding.sh
git commit -m "feat: add port forwarding test script"
```

---
## Task 8: Update Project Documentation

**Files:**
- Modify: `/Users/ntg/Documents/Personal_Projects/medgemma_RAG/CLAUDE.md`

**Step 1: Find SSH configuration section in CLAUDE.md**

Run: `grep -n "SSH\|ssh" /Users/ntg/Documents/Personal_Projects/medgemma_RAG/CLAUDE.md | head -5`
Expected: Shows lines mentioning SSH

**Step 2: Add port forwarding section**

Add to CLAUDE.md in appropriate section (likely under "Remote Mode" or "Deployment"):
```bash
# Find a good insertion point after SSH configuration
INSERT_LINE=$(grep -n "Configure SSH" /Users/ntg/Documents/Personal_Projects/medgemma_RAG/CLAUDE.md | head -1 | cut -d: -f1)
if [ -n "$INSERT_LINE" ]; then
    sed -i '' "${INSERT_LINE}a\\
### Port Forwarding for Local Access\\
Added to \`~/.ssh/config\` for \`medgemma-gpu\` host:\\
\`\`\`ssh\\
LocalForward 8888 localhost:8888    # Jupyter Notebook\\
LocalForward 7860 localhost:7860    # Gradio UI\\
\`\`\`\\
Access locally: \`http://localhost:8888\` (Jupyter) and \`http://localhost:7860\` (Gradio)\\
See \`docs/ssh-port-forwarding-guide.md\` for full usage instructions." /Users/ntg/Documents/Personal_Projects/medgemma_RAG/CLAUDE.md
fi
```

**Step 3: Verify CLAUDE.md update**

Run: `grep -A 5 "Port Forwarding for Local Access" /Users/ntg/Documents/Personal_Projects/medgemma_RAG/CLAUDE.md`
Expected: Shows new port forwarding section

**Step 4: Commit documentation update**

```bash
git add /Users/ntg/Documents/Personal_Projects/medgemma_RAG/CLAUDE.md
git commit -m "docs: add SSH port forwarding to project documentation"
```

---
## Task 9: Final Integration Test

**Files:**
- No file changes, end-to-end testing

**Step 1: Verify all components**

Run: `ls -la /Users/ntg/Documents/Personal_Projects/medgemma_RAG/docs/ssh-port-forwarding-guide.md /Users/ntg/Documents/Personal_Projects/medgemma_RAG/scripts/test-port-forwarding.sh`
Expected: Both files exist with correct permissions

**Step 2: Test SSH config syntax**

Run: `ssh -G medgemma-gpu 2>&1 | grep -E "hostname|localforward" | head -5`
Expected: Shows hostname and localforward configurations

**Step 3: Simulate workflow**

```bash
echo "Simulated workflow:"
echo "1. Start EC2: ./scripts/ec2-start.sh"
echo "2. Check status: ./scripts/ec2-status.sh"
echo "3. Update ~/.ssh/config if IP changed"
echo "4. Connect: ssh medgemma-gpu"
echo "5. On EC2: start services"
echo "6. Locally: access http://localhost:8888 and http://localhost:7860"
```

**Step 4: Run comprehensive test**

Run: `/Users/ntg/Documents/Personal_Projects/medgemma_RAG/scripts/test-port-forwarding.sh`
Expected: All tests pass or show actionable information

---
## Task 10: Final Commit and Summary

**Files:**
- All modified files

**Step 1: Check git status**

Run: `cd /Users/ntg/Documents/Personal_Projects/medgemma_RAG && git status`
Expected: Shows modified files ready for commit

**Step 2: Create final summary commit**

```bash
cd /Users/ntg/Documents/Personal_Projects/medgemma_RAG
git add -u
git commit -m "feat: complete SSH port forwarding implementation

- Added LocalForward directives to ~/.ssh/config for Jupyter (8888) and Gradio (7860)
- Created usage guide: docs/ssh-port-forwarding-guide.md
- Added test script: scripts/test-port-forwarding.sh
- Updated project documentation in CLAUDE.md
- Verified port availability and SSH configuration

Enables local browser access to services running on EC2 GPU instance:
- Jupyter Notebook: http://localhost:8888
- Gradio UI: http://localhost:7860"
```

**Step 3: Verify final state**

Run: `git log --oneline -3`
Expected: Shows recent commits including port forwarding implementation

**Step 4: Output completion message**

```bash
echo "═══════════════════════════════════════"
echo "SSH Port Forwarding Implementation Complete"
echo "═══════════════════════════════════════"
echo ""
echo "What was implemented:"
echo "1. ✓ Added port forwarding to ~/.ssh/config"
echo "2. ✓ Created usage documentation"
echo "3. ✓ Added test script"
echo "4. ✓ Updated project docs"
echo ""
echo "To use:"
echo "1. Start EC2: ./scripts/ec2-start.sh"
echo "2. Update ~/.ssh/config with new IP if changed"
echo "3. Connect: ssh medgemma-gpu"
echo "4. Start services on EC2"
echo "5. Access locally: http://localhost:8888 (Jupyter)"
echo "6. Access locally: http://localhost:7860 (Gradio)"
echo ""
echo "Test with: ./scripts/test-port-forwarding.sh"
echo "Guide: docs/ssh-port-forwarding-guide.md"
echo "═══════════════════════════════════════"
```

---
## Execution Options

Plan complete and saved to `docs/plans/2026-02-14-ssh-port-forwarding-design.md`.

**Two execution options:**

1. **Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

2. **Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

**Which approach?**SSH config backup created: /Users/ntg/.ssh/config.backup.20260214_111101
