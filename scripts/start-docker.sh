docker run -it --rm \
  --network=host \
  --device=/dev/kfd --device=/dev/dri \
  --group-add=video --group-add=render \
  --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --shm-size 8G \
  -v /workspace/.cache/huggingface:/root/.cache/huggingface \
  -v /workspace/AIAC:/workspace/AIAC \
  -w /workspace/AIAC \
  --name rocm-jupyter \
  edaamd/aiac:latest

