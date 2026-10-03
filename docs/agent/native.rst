Native Backend (macOS, Metal)
=============================

With ``backend = "native"`` the agent runs each kernel as a host process tree instead of a container.
On an Apple Silicon Mac the kernel reaches the Apple GPU through Metal, which no Linux container on macOS can.

Use it on single-tenant development agents only.
The design is in `BEP-1079 <https://github.com/lablup/backend.ai/blob/main/proposals/BEP-1079-native-process-agent-backend.md>`_.

What you get
------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Item
     - Behavior
   * - Batch session
     - The startup command runs on the host with the image's Python environment first in ``PATH``.
   * - Interactive session
     - Services the kernel runner starts itself (``jupyter``, preopen ports) work. ``sshd`` and ``ttyd`` are not available.
   * - Inference session
     - A deployment serves a model vfolder, for example with the ``mlx-lm`` runtime variant.
   * - ``metal.device`` slot
     - One device per host, reported by the ``ai.backend.accelerator.metal`` plugin.
   * - Agent restart
     - Running kernels keep running and are re-adopted when the agent starts again.

Limits
------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Limit
     - Consequence
   * - No isolation
     - A kernel runs as the agent's OS user and sees that user's files, network and devices.
   * - No resource enforcement
     - ``cpu``, ``mem`` and ``metal.device`` are accounted, not limited. Any host process can use the GPU.
   * - Unified memory
     - The GPU working set and ``mem`` are the same RAM.
   * - No ``/home/work``
     - The home directory is ``<scratch-root>/<kernel-id>/work``. Address vfolders relative to the home directory.
   * - Mounts outside ``/home/work``
     - A mount at ``/models`` is the symlink ``<scratch-root>/<kernel-id>/mounts/models``. The model path of an inference session is rewritten to it.
   * - Kernel paths in environment variables
     - A value that is a mount's kernel path, or a path under one, is rewritten to the host path (``DATASET_FILE=/home/work/data/train.jsonl`` reaches the process as the host path). ``BACKENDAI_PERSISTENT_PATHS`` is rewritten element-wise. Every mount also sets ``BACKENDAI_MOUNT_<NAME>`` to its host path, ``<NAME>`` being the last path component upper-cased with non-alphanumerics as ``_``. Paths inside commands are not rewritten.
   * - Read-only mounts
     - Not enforced; a mount is a symlink to the vfolder host path.
   * - Service ports
     - The host port equals the declared port. A port another kernel declares, or one with a listener on the host, fails the kernel creation.
   * - Scheduling
     - The scheduler matches the CPU architecture only. Keep native agents in their own resource group.

Prerequisites
-------------

- An Apple Silicon Mac.
- A development install with the manager, storage-proxy, app-proxy coordinator and app-proxy worker running, and the database at the alembic head.
- A Docker daemon on the manager host. The ``local`` container registry lists images from it.
- The ``local`` container registry row (``fixtures/manager/example-container-registries-local.json``).
- `uv <https://docs.astral.sh/uv/>`_.
- ``./bai`` logged in as a superadmin.

1. Create the Python environment
--------------------------------

.. code-block:: bash

   ENV_DIR=$HOME/.local/share/backend.ai/native-envs/mlx-lm
   uv venv "$ENV_DIR" --python 3.12
   uv pip install --python "$ENV_DIR/bin/python" mlx mlx-lm
   "$ENV_DIR/bin/python" -c "import mlx.core as mx; print(mx.default_device(), mx.metal.is_available())"

The last command prints ``Device(gpu, 0) True``.

2. Register the image
---------------------

The image is metadata only.
The agent reads its labels and never runs it; ``ai.backend.runtime-path`` names the interpreter on the agent host.

.. literalinclude:: ../../configs/agent/native-image-stub.dockerfile
   :language: dockerfile

.. code-block:: bash

   docker build --platform linux/arm64 \
     -f configs/agent/native-image-stub.dockerfile \
     --build-arg RUNTIME_PATH="$ENV_DIR/bin/python" \
     -t stable/mlx-lm:0.32-macos configs/agent
   ./bai gql 'mutation { rescan_images(registry: "local") { ok msg task_id } }'
   ./bai admin image search --name-contains mlx-lm

The search lists ``local/stable/mlx-lm:0.32-macos`` with architecture ``aarch64``. Note its ``id``.

``./backend.ai mgr image rescan local`` fails with a SQLAlchemy mapper error; use the GraphQL mutation.

3. Register the ``metal.device`` slot type
------------------------------------------

The manager reports a slot only when its type is registered.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Install
     - Command
   * - New
     - ``./backend.ai mgr fixture populate fixtures/manager/example-resource-slot-types.json``
   * - Existing
     - ``./bai resource-slot slot-type create metal.device count --display-name "Apple GPU" --display-unit GPU``

.. code-block:: bash

   ./bai resource-slot slot-type search --limit 50

4. Configure the agent
----------------------

.. code-block:: bash

   cp configs/agent/halfstack-native.toml agent.toml

``configs/agent/halfstack-native.toml`` is ``halfstack.toml`` with three settings changed:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Setting
     - Value
   * - ``[agent] backend``
     - ``"native"``
   * - ``[agent] allow-compute-plugins``
     - ``["ai.backend.accelerator.metal"]``
   * - ``[resource] allocation-order``
     - ``["metal", "cpu", "mem"]``

Adapt the etcd address and namespace, the ports, and the paths to your install.
The sample uses the ports and paths of ``halfstack.toml``; stop the Docker agent before starting this one.

5. Start the agent
------------------

.. code-block:: bash

   ./backend.ai ag start-server -f agent.toml --debug
   ./bai admin agent search --limit 5

- The agent log contains ``native backend selected without isolation or resource enforcement``.
- The agent's ``resource_info.capacity`` contains ``"metal.device": "1"``.

6. Run a batch session
----------------------

``session.json``, with the image ``id`` from step 2 and a project ``id`` from ``./bai admin project search``:

.. code-block:: json

   {
     "session_name": "metal-check",
     "session_type": "batch",
     "image_id": "<image-id>",
     "project_id": "<project-id>",
     "resource_entries": [
       {"resource_type": "cpu", "quantity": "2"},
       {"resource_type": "mem", "quantity": "4g"},
       {"resource_type": "metal.device", "quantity": "1"}
     ],
     "batch": {
       "startup_command": "python -c \"import mlx.core as mx; print(mx.default_device(), mx.metal.is_available())\""
     }
   }

.. code-block:: bash

   ./bai session enqueue @session.json
   ./bai session get <session-id>
   ./bai session logs <session-id>

The session ends as ``TERMINATED`` with result ``success``, and the log contains:

.. code-block:: text

   Device(gpu, 0) True

7. Serve a model
----------------

Create a model vfolder and note its ``id``:

.. code-block:: bash

   ./bai vfolder create --name mlx-model --usage-mode model

Fill it from a batch session. The vfolder is mounted in the home directory under its name. ``download.json``:

.. code-block:: json

   {
     "session_name": "model-download",
     "session_type": "batch",
     "image_id": "<image-id>",
     "project_id": "<project-id>",
     "resource_entries": [
       {"resource_type": "cpu", "quantity": "2"},
       {"resource_type": "mem", "quantity": "4g"}
     ],
     "mounts": [{"vfolder_id": "<vfolder-id>"}],
     "batch": {
       "startup_command": "hf download mlx-community/Llama-3.2-1B-Instruct-4bit --local-dir mlx-model"
     }
   }

.. code-block:: bash

   ./bai session enqueue @download.json

``revision.json``, with the ``id`` of the ``mlx-lm`` variant from ``./bai runtime-variant search --name-contains mlx-lm``:

.. code-block:: json

   {
     "cluster_config": {"mode": "single-node", "size": 1},
     "resource_config": {"resource_slots": {"entries": [
       {"resource_type": "cpu", "quantity": "2"},
       {"resource_type": "mem", "quantity": "4g"},
       {"resource_type": "metal.device", "quantity": "1"}
     ]}},
     "image": {"id": "<image-id>"},
     "model_runtime_config": {"runtime_variant_id": "<runtime-variant-id>"},
     "model_mount_config": {"vfolder_id": "<vfolder-id>", "mount_destination": "/models"},
     "auto_activate": true
   }

.. code-block:: bash

   ./bai deployment create --name mlx-lm-demo --project-id <project-id> \
     --resource-group default --replicas 1 --initial-revision @revision.json
   ./bai deployment get <deployment-id>
   ./bai deployment replica search <deployment-id>
   ./bai deployment access-token create <deployment-id> --expires-at 2026-12-31
   curl -s <endpoint-url>v1/chat/completions \
     -H "Authorization: BackendAI <token>" -H 'Content-Type: application/json' \
     -d '{"messages": [{"role": "user", "content": "What is the capital of France?"}], "max_tokens": 32}'
   ./bai deployment delete <deployment-id>

- ``deployment get`` shows ``READY`` and the ``endpoint_url`` once the replica is ``running``, ``active``, ``healthy``. This took four minutes on an M5 Pro.
- The reply contains ``The capital of France is Paris.``
- After the delete, no ``bai-krunner`` process and no directory under the scratch root remain.

Where a kernel's files are
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Path
     - Content
   * - ``<scratch-root>/<kernel-id>/work``
     - Home and working directory; vfolders mounted under ``/home/work`` are symlinks here
   * - ``<scratch-root>/<kernel-id>/mounts``
     - Symlinks for mounts outside ``/home/work``
   * - ``<scratch-root>/<kernel-id>/config/kernel.log``
     - Output of the kernel runner and the user processes
   * - ``<scratch-root>/<kernel-id>/config/native-kernel.json``
     - Process id, ports and labels the agent uses to find the kernel again

Troubleshooting
---------------

.. list-table::
   :header-rows: 1
   :widths: 35 30 35

   * - Symptom
     - Cause
     - Fix
   * - The agent does not report ``metal.device``
     - The slot type is not registered
     - Step 3
   * - Kernel creation fails with ``ValueError`` for a session requesting ``metal.device``
     - ``allocation-order`` does not list ``metal``
     - Step 4
   * - ``KernelRuntimeNotFoundError``; the session stays in ``CREATING``
     - ``ai.backend.runtime-path`` is not an executable on the agent host
     - Rebuild the image with the right ``RUNTIME_PATH``, rescan, terminate the session
   * - ``PortConflictError: Service ports already in use on the agent host``
     - Another kernel or host process holds the declared port
     - Free the port, or declare another one
   * - Replicas of a deployment are replaced before they become healthy
     - The health check ``initial_delay`` is shorter than the model load
     - Raise ``initial_delay`` in the model definition
   * - The model server listens on every interface
     - The ``mlx-lm`` variant starts with ``--host 0.0.0.0`` and a host process has no network namespace
     - Use the ``custom`` variant with a model definition whose command binds ``127.0.0.1``
   * - A session ends with result ``failure`` and ``self-terminated``
     - The kernel runner process died or was killed outside the agent
     - Read ``kernel.log`` before the session is cleaned, or ``./bai session logs``
