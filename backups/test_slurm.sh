#!/bin/bash
#SBATCH --account=project_2004072
#SBATCH --job-name=test
#SBATCH --output=/scratch/project_2004072/ImACCESS/trash/logs/%x.out
#SBATCH --mail-user=farid.alijani@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=40G
#SBATCH --partition=test
#SBATCH --time=00-00:15:00

set -euo pipefail

echo "=========================================="
echo "Cluster: $SLURM_CLUSTER_NAME"
echo "Node: $SLURM_NODELIST"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="

echo -e "\n=== Environment Variables ==="
echo "TMPDIR: ${TMPDIR:-NOT SET}"
echo "LOCAL_SCRATCH: ${LOCAL_SCRATCH:-NOT SET}"

echo -e "\n=== Checking /tmp ==="
df -h /tmp
ls -la /tmp/ | head -20

echo -e "\n=== Checking for /local_scratch ==="
if [ -d "/local_scratch" ]; then
    df -h /local_scratch
    ls -la /local_scratch/ | head -20
else
    echo "/local_scratch does not exist"
fi

echo -e "\n=== Checking for /nvme ==="
if [ -d "/nvme" ]; then
    df -h /nvme
    ls -la /nvme/ | head -20
else
    echo "/nvme does not exist"
fi

echo -e "\n=== All mounted filesystems ==="
df -h | grep -E "(Filesystem|nvme|local|tmp|scratch)"

echo -e "\n=== NVMe devices ==="
lsblk | grep -i nvme || echo "No NVMe devices visible"

echo -e "\n=== I/O Performance Tests ==="
echo "Testing /tmp:"
dd if=/dev/zero of=/tmp/test_${SLURM_JOB_ID}.bin bs=1M count=100 2>&1 | tail -1
rm -f /tmp/test_${SLURM_JOB_ID}.bin

echo -e "\nTesting /scratch:"
dd if=/dev/zero of=/scratch/project_2004072/test_${SLURM_JOB_ID}.bin bs=1M count=100 2>&1 | tail -1
rm -f /scratch/project_2004072/test_${SLURM_JOB_ID}.bin

if [ -d "/local_scratch" ]; then
    echo -e "\nTesting /local_scratch:"
    mkdir -p /local_scratch/$SLURM_JOB_ID
    dd if=/dev/zero of=/local_scratch/$SLURM_JOB_ID/test.bin bs=1M count=100 2>&1 | tail -1
    rm -f /local_scratch/$SLURM_JOB_ID/test.bin
fi

echo -e "\n=========================================="
echo "Diagnostic complete"
echo "=========================================="