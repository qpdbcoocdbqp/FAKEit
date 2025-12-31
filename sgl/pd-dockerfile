FROM lmsysorg/sglang:v0.5.6.post2-cu130-runtime AS stage
RUN git clone https://github.com/kvcache-ai/Mooncake --recursive && \
    cd Mooncake && \
    bash dependencies.sh -y && \
    export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cuda/lib64 && \
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64 && \
    export CUDA_HOME=/usr/local/cuda-13 && \
    mkdir build && cd build && \
    cmake .. -DBUILD_UNIT_TESTS=OFF -DUSE_CUDA=ON -DWITH_STORE=OFF -DCMAKE_BUILD_TYPE=Release && \
    make -j 4 && make install && \
    python3 -c 'from mooncake.engine import TransferEngine;print("import success");'
ENTRYPOINT []
