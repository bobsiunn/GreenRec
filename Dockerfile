FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

SHELL ["/bin/bash", "-c"]

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV HOME /home
RUN mkdir -p ${HOME}
WORKDIR ${HOME}

RUN rm -rf /opt/conda

RUN apt-get update --fix-missing
RUN apt-get install -y wget bzip2 curl git python3-pip vim unzip
RUN DEBIAN_FRONTEND=noninteractive apt install -yq build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev libpython3-dev libdbus-1-dev libsystemd-dev libglib2.0-dev pkg-config
RUN apt-get clean && rm -rf /var/lib/apt/lists/*


RUN git config --global user.name "bob.siunn"
RUN git config --global user.email "psh000701@gmail.com"
RUN git config --global credential.helper store

RUN echo "alias ll='ls --color=auto -alF'" >> ~/.bashrc
RUN echo "PS1='\[\e[01;32m\]\u\[\e[01;37m\]@\[\e[01;33m\]\H\[\e[01;37m\]:\[\e[01;32m\]\w\[\e[01;37m\]\$\[\033[0;37m\]'" >> ~/.bashrc
RUN echo "if [ -x /usr/bin/dircolors ]; then" >> ~/.bashrc
RUN echo -e "\ttest -r ~/.dircolors && eval "$(dircolors -b ~/.dircolors)" || eval "$(dircolors -b)"" >> ~/.bashrc
RUN echo -e "\talias ls='ls --color=auto'" >> ~/.bashrc
RUN echo -e "\talias grep='grep --color=auto'" >> ~/.bashrc
RUN echo -e "\talias fgrep='fgrep --color=auto'" >> ~/.bashrc
RUN echo -e "\talias egrep='egrep --color=auto'\nfi" >> ~/.bashrc

RUN wget https://www.python.org/ftp/python/3.8.1/Python-3.8.1.tgz && \
    tar -xzf Python-3.8.1.tgz && \
    cd Python-3.8.1 && \
    ./configure --enable-optimizations && \
    make -j"$(nproc)" && \
    make altinstall

RUN ln -sf /usr/local/bin/python3.8 /usr/bin/python && \
    ln -sf /usr/local/bin/pip3.8 /usr/bin/pip

RUN ln -sf /usr/local/bin/python3.8 /usr/bin/python3 && \
    ln -sf /usr/local/bin/pip3.8 /usr/bin/pip3

RUN pip install pip==20.0.2

CMD [ "/bin/bash" ]