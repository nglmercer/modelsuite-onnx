"""modelsuite: discover, analyze, convert, optimize, validate and benchmark ML models.

The package root stays import-light on purpose (no torch / onnx / transformers); every heavy
dependency is imported lazily by the module that needs it.
"""

__version__ = "0.1.0"
