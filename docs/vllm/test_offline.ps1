# 1. Build the Docker image
# This stage requires an internet connection to download all dependencies.
# It uses AlmaLinux 8 (RHEL 8 compatible) to prepare the environment.
Write-Host "Building vLLM Offline Test Image..." -ForegroundColor Cyan
docker build -t vllm-offline-test -f Dockerfile.offline_test .

# 2. Run the offline container
# The --network none flag ensures the container has absolutely NO internet access,
# guaranteeing that your offline installation process is strictly tested.
Write-Host "Starting the offline container..." -ForegroundColor Cyan
Write-Host "Inside the container, you can proceed with Phase 3 of your guideline.md" -ForegroundColor Yellow
docker run -it --rm --network none --name vllm-offline-env vllm-offline-test
