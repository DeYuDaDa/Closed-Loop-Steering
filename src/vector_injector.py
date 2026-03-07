"""
Vector Injector — Control Vector Loading & Management
========================================================
Loads CAA-extracted (and optionally PCA-purified) control vectors from disk,
normalises them, and serves them in the shape expected by the steering hook.

Used by run_experiment.py to supply the control vector to the closed-loop
steering pipeline.
"""

import os
import torch
from typing import Optional


class VectorInjector:
    """
    Manages loading, normalising, and serving control vectors.

    Usage:
        injector = VectorInjector("./vectors/qwen3-8b", device="cuda", model_dtype=torch.bfloat16)
        if injector.activate("critic", coeff=1.0):
            v = injector.get_normalized_vector()   # shape [1, 1, d]
            ...
        injector.deactivate()
    """

    def __init__(
        self,
        vector_dir: str,
        device: str = "cpu",
        model_dtype: torch.dtype = torch.float32,
    ):
        """
        Args:
            vector_dir: Directory where .pt vector files are stored.
            device: Target device for the loaded tensors.
            model_dtype: dtype to cast the vector to (should match model weights).
        """
        self.vector_dir = vector_dir
        self.device = device
        self.model_dtype = model_dtype

        # Internal state
        self._active_vector: Optional[torch.Tensor] = None
        self._active_name: Optional[str] = None
        self._coeff: float = 1.0

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def activate(self, name: str, coeff: float = 1.0) -> bool:
        """
        Load a control vector by name and mark it as active.

        Resolution order:
            1. {vector_dir}/{name}.pt           (PCA-purified, preferred)
            2. {vector_dir}/{name}_raw.pt       (raw CAA vector, fallback)

        Args:
            name:  Logical name of the vector (e.g. "critic").
            coeff: Scaling coefficient applied during normalisation.

        Returns:
            True if the vector was loaded successfully, False otherwise.
        """
        purified_path = os.path.join(self.vector_dir, f"{name}.pt")
        raw_path = os.path.join(self.vector_dir, f"{name}_raw.pt")

        load_path: Optional[str] = None
        if os.path.isfile(purified_path):
            load_path = purified_path
        elif os.path.isfile(raw_path):
            load_path = raw_path
            print(f"[VectorInjector] ⚠️  Purified vector not found, "
                  f"falling back to raw: {raw_path}")

        if load_path is None:
            print(f"[VectorInjector] ❌ No vector file found for '{name}' "
                  f"in {self.vector_dir}")
            return False

        try:
            v = torch.load(load_path, map_location="cpu", weights_only=True)
            self._active_vector = v.to(device=self.device, dtype=self.model_dtype)
            self._active_name = name
            self._coeff = coeff
            print(f"[VectorInjector] ✅ Loaded '{name}' from {load_path}")
            print(f"  shape={list(v.shape)}, "
                  f"norm={v.float().view(-1).norm().item():.4f}, coeff={coeff}")
            return True
        except Exception as e:
            print(f"[VectorInjector] ❌ Failed to load {load_path}: {e}")
            return False

    def get_normalized_vector(self) -> torch.Tensor:
        """
        Return the active vector, L2-normalised and shaped as [1, 1, d].

        The coefficient is applied *after* normalisation:
            v_out = coeff * (v / ||v||)

        This ensures the steering hook receives a unit-direction vector,
        with magnitude controlled by coeff if needed.
        """
        if self._active_vector is None:
            raise RuntimeError(
                "No active vector. Call .activate() first."
            )

        v = self._active_vector.clone()

        # Flatten → normalise → restore
        v_flat = v.view(-1).float()
        v_flat = v_flat / v_flat.norm()
        v_flat = v_flat * self._coeff
        v_out = v_flat.to(dtype=self.model_dtype)

        # Reshape to [1, 1, d]
        v_out = v_out.view(1, 1, -1)
        return v_out.to(device=self.device)

    def get_raw_norm(self) -> float:
        """Return the L2 norm of the loaded vector before normalization.

        Useful for coeff calibration: the raw norm reflects the
        vector's magnitude relative to the layer's activation space.
        """
        if self._active_vector is None:
            raise RuntimeError("No active vector. Call .activate() first.")
        return self._active_vector.float().view(-1).norm().item()

    def deactivate(self):
        """Release the active vector from memory."""
        self._active_vector = None
        self._active_name = None
        self._coeff = 1.0

    # ------------------------------------------------------------------ #
    #  Utility helpers                                                    #
    # ------------------------------------------------------------------ #

    def list_available(self) -> list[str]:
        """List all vector names available in vector_dir."""
        if not os.path.isdir(self.vector_dir):
            return []
        names = set()
        for f in os.listdir(self.vector_dir):
            if f.endswith(".pt"):
                # "critic.pt" → "critic",  "critic_raw.pt" → "critic"
                base = f.replace("_raw.pt", ".pt").replace(".pt", "")
                names.add(base)
        return sorted(names)

    @property
    def is_active(self) -> bool:
        return self._active_vector is not None

    @property
    def active_name(self) -> Optional[str]:
        return self._active_name

    def __repr__(self):
        status = f"active='{self._active_name}'" if self.is_active else "inactive"
        return f"VectorInjector(dir='{self.vector_dir}', {status})"
