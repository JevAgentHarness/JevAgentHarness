"""Put on PYTHONPATH (``launch.run`` / the box's docker helper) so EVERY Python process of the Miles job -- the Ray
trainer ranks included -- installs the trainer key patch at start-up (``keys_patch``: Miles' FSDP actor then hands
``batch["metadata"]`` to the decision loss). Only registers an import hook; it imports nothing heavy."""
try:
    from jevharness.training.miles_port import keys_patch as _jk

    _jk.install()
except Exception:  # noqa: BLE001 - never break an unrelated interpreter
    pass
