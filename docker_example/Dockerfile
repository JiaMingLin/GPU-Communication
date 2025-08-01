FROM python:3.10-slim

# Install dependencies
RUN pip install numpy

# Set working directory
WORKDIR /app

# Add entrypoint wrapper script
RUN echo '#!/bin/bash\n' \
         'echo "==> Listing /app contents:"\n' \
         'ls -alh /app\n' \
         'echo ""\n' \
         'if [[ ! -x "/app/start.sh" ]]; then\n' \
         '  echo "❌ ERROR: start.sh not found or not executable in /app!" >&2\n' \
         '  exit 1\n' \
         'fi\n' \
         'echo "🚀 Launching /app/start.sh..."\n' \
         '/app/start.sh' > /entrypoint.sh && chmod +x /entrypoint.sh

# Set custom entrypoint
ENTRYPOINT ["/entrypoint.sh"]