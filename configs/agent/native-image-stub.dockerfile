# Label-only image for the native agent backend. The agent never runs it; it reads the labels.
FROM scratch
ARG RUNTIME_PATH
LABEL ai.backend.kernelspec="1" \
      ai.backend.features="batch query" \
      ai.backend.role="COMPUTE" \
      ai.backend.base-distro="macos" \
      ai.backend.runtime-type="python" \
      ai.backend.runtime-path="${RUNTIME_PATH}" \
      ai.backend.resource.min.cpu="1" \
      ai.backend.resource.min.mem="1g" \
      ai.backend.service-ports=""
