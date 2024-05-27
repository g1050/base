docker run -i -t --gpus all \
    --name xkgao_torch \
    -v /data:/data \
    -p 10833:22 \
    docker.io/nvidia/cuda:11.8.0-base-ubuntu18.04 \
    /bin/bash
