FROM phusion/baseimage:jammy-1.0.1

ENV DEBIAN_FRONTEND=noninteractive
ENV term=xterm
ARG CLANG_VERSION=13
ARG ROCM_VERSION=5.3

ENV TZ=Europe/London
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# oneapi
#    ONEAPI_PACKAGE=l_BaseKit_p_2023.1.0.46401_offline.sh
#    QUARTUS_PACKAGE=QuartusLiteSetup-22.1std.1.917-linux.run
COPY --from=artisan-support:oneapi /opt/intel/ /opt/intel/

RUN echo "deb [arch=amd64] http://downloads.skewed.de/apt jammy main" >> /etc/apt/sources.list && \
    curl -fsSl https://keys.openpgp.org/vks/v1/by-fingerprint/793CEFE14DBC851A2BFB1222612DEFB798507F25 | gpg --dearmor --yes --output /etc/apt/trusted.gpg.d/graph-tool.gpg && \
    apt-get update && apt-get install -y --no-install-recommends \
          wget \
          python3 \
          python3-distutils \
          libpython3-dev \
          git \
          build-essential \
          nano \
          gnupg \
          curl \
          zip \
          telnet \
          ca-certificates \
          sudo \
          python3-graph-tool \
          libzmq3-dev \
          libpango-1.0-0 \
          libpangoft2-1.0-0 \
          libboost-dev \
          libncurses5 \
          && rm -rf /var/lib/apt/lists/*

# # ROCM
RUN wget https://repo.radeon.com/amdgpu-install/5.3/ubuntu/focal/amdgpu-install_5.3.50300-1_all.deb -O /tmp/amdgpu.deb && \
    dpkg -i /tmp/amdgpu.deb && apt update && \
    apt install -y rocm-dev rocm-libs \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /tmp/amdgpu.deb


# setting up python and pip3 packages
 RUN  curl -sS https://bootstrap.pypa.io/pip/get-pip.py | python3 && \
    ln -s /usr/local/bin/pip3 /usr/bin/pip3 && ln -s /usr/local/bin/pip3 /usr/bin/pip && \
    pip3 install --upgrade pip && \
    pip3 install --ignore-installed blinker && \
    rm -rf /usr/lib/python3/dist-packages/numpy* /usr/lib/python3/dist-packages/scipy* && \
    pip3 install --upgrade pip \
                           virtualenv \
                           setuptools \
                           pytest \
                           cached_property \
                           colorama \
                           Pygments \
                           matplotlib \
                           scipy \
                           numpy \
                           pyparsing \
                           flask \
                           flask-classful \
                           graphviz \
                           pylint \
                           dill \
                           cython \
                           lark-parser \
                           tabulate \
                           --no-binary=pyzmq pyzmq \
                           pcpp \
                           lxml \
                           pyyaml \
                           islpy

# patch graph-tool
RUN sed -i 's/from graph_tool.draw import \*/# from graph_tool.draw import \*/' /usr/lib/python3/dist-packages/graph_tool/all.py && \
    sed -i 's/import graph_tool.draw/pass/' /usr/lib/python3/dist-packages/graph_tool/all.py

# install clang
RUN echo "deb https://apt.llvm.org/jammy/ llvm-toolchain-jammy-$CLANG_VERSION main" >> /etc/apt/sources.list && \
    echo "deb-src https://apt.llvm.org/jammy/ llvm-toolchain-jammy-$CLANG_VERSION main" >> /etc/apt/sources.list && \
    wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key|  gpg --dearmor --yes --output /etc/apt/trusted.gpg.d/llvm.gpg && \
    apt update && \
    apt-get install -y libllvm-$CLANG_VERSION-ocaml-dev libllvm${CLANG_VERSION} llvm-$CLANG_VERSION llvm-$CLANG_VERSION-dev llvm-$CLANG_VERSION-doc llvm-$CLANG_VERSION-examples llvm-$CLANG_VERSION-runtime && \
    apt-get install -y libomp-$CLANG_VERSION-dev clang-$CLANG_VERSION clang-tools-$CLANG_VERSION clang-$CLANG_VERSION-doc libclang-common-$CLANG_VERSION-dev libclang-$CLANG_VERSION-dev libclang1-$CLANG_VERSION clang-format-$CLANG_VERSION python3-clang-$CLANG_VERSION clangd-$CLANG_VERSION && \
    rm -rf /var/lib/apt/lists/* && \
    ln -s /usr/bin/clang-$CLANG_VERSION /usr/bin/clang && \
    ln -s /usr/bin/clang++-$CLANG_VERSION /usr/bin/clang++ && \
    ln -s /usr/bin/clang-format-$CLANG_VERSION /usr/bin/clang-format

#scripts
COPY entrypoint /etc/artisan/
COPY startup /etc/artisan/
COPY exports/ /opt/

RUN chmod +x /etc/artisan/entrypoint && \
    echo "source /etc/artisan/startup" >> /etc/bash.bashrc && \
    ln -s /usr/bin/python3 /usr/bin/artisan && \
    sed -i 's/^%sudo.*$/%sudo   ALL=(ALL:ALL) NOPASSWD:ALL/' /etc/sudoers && \
    sudo chmod a=u /root -R


WORKDIR /workspace

ENTRYPOINT ["/etc/artisan/entrypoint"]





