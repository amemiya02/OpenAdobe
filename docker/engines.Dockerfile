# OpenAdobe engine image: every engine the control layer drives runs here, version-locked.
# Build: docker build -f docker/engines.Dockerfile -t openadobe/engines:dev docker/
FROM debian:trixie-slim

ENV DEBIAN_FRONTEND=noninteractive LC_ALL=C.UTF-8
RUN apt-get update && apt-get install -y --no-install-recommends \
        melt libmlt-data ffmpeg frei0r-plugins swh-plugins sox \
        fonts-noto-cjk fonts-noto-cjk-extra fonts-dejavu-core fonts-roboto fontconfig ca-certificates curl \
        ardour \
        blender \
        xvfb xauth \
    && rm -rf /var/lib/apt/lists/* \
    && fc-cache -f

# RNNoise models for ffmpeg's arnndn (speech denoise); their sha256 is recorded in the lock file.
RUN mkdir -p /opt/openadobe/rnnn && cd /opt/openadobe/rnnn \
    && for m in cb mp sh bd lq; do \
         curl -fsSL -o $m.rnnn "https://raw.githubusercontent.com/richardpl/arnndn-models/master/$m.rnnn" || exit 1; \
       done

# Record exact engine versions so renders are attributable (see `openadobe doctor`).
RUN { echo "melt=$(dpkg-query -W -f='${Version}' melt)"; \
      echo "ffmpeg=$(dpkg-query -W -f='${Version}' ffmpeg)"; \
      echo "frei0r=$(dpkg-query -W -f='${Version}' frei0r-plugins)"; \
      echo "ardour=$(dpkg-query -W -f='${Version}' ardour)"; \
      echo "blender=$(dpkg-query -W -f='${Version}' blender)"; \
      for m in /opt/openadobe/rnnn/*.rnnn; do echo "rnnn.$(basename $m .rnnn)=$(sha256sum $m | cut -c1-16)"; done; \
    } > /etc/openadobe-engines.lock

WORKDIR /work
