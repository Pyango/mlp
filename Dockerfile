FROM continuumio/miniconda3:4.9.2-alpine

RUN apk add --no-cache \
    wget \
    bzip2 \
    git \
    openssh \
    gcc \
    graphviz \
    ttf-dejavu \
    musl-dev


RUN mkdir /var/run/sshd \
  && echo 'root:screencast' | /usr/sbin/chpasswd \
  && sed -i '/PermitRootLogin/c\PermitRootLogin yes' /etc/ssh/sshd_config \
  && sed -i '/AllowTcpForwarding/c\AllowTcpForwarding yes' /etc/ssh/sshd_config \
  # SSH login fix. Otherwise user is kicked off after login
  && sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /usr/sbin/sshd \
  && echo "export VISIBLE=now" >> /etc/profile \
  && ssh-keygen -A
ENV NOTVISIBLE "in users profile"

RUN mkdir -p /app

COPY ./requirements.yml /app/requirements.yml
RUN /opt/conda/bin/conda env update --file /app/requirements.yml --prune \
  && /opt/conda/bin/conda clean -ay \
  && find /opt/conda/ -follow -type f -name '*.a' -delete \
  && find /opt/conda/ -follow -type f -name '*.pyc' -delete \
  && find /opt/conda/ -follow -type f -name '*.js.map' -delete

ENV PATH="/opt/conda/bin:${PATH}"
