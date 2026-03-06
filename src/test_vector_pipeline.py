"""
Test Vector Pipeline — End-to-End Verification (No GPU required)
===================================================================
Verifies the full vector extraction → save → load → read cycle using
simulated tensors, so this can run on any machine without a model.

Usage:
    cd /path/to/src
    python test_vector_pipeline.py

All checks should print PASS.
"""

import os
import sys
import shutil
import tempfile
import torch
import numpy as np

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from vector_injector import VectorInjector
from manifold_utils import ManifoldProjector
from spherical_injector import spherical_rotate


def test_save_load_cycle():
    """Test: save a vector to .pt, load via VectorInjector, values match."""
    print("\n" + "="*60)
    print("  TEST 1: Save/Load Cycle")
    print("="*60)

    tmpdir = tempfile.mkdtemp(prefix="test_vectors_")
    try:
        d = 4096  # typical hidden dim
        v_original = torch.randn(d)

        # Save as purified vector
        torch.save(v_original, os.path.join(tmpdir, "critic.pt"))

        # Load via VectorInjector
        injector = VectorInjector(tmpdir, device="cpu", model_dtype=torch.float32)
        success = injector.activate("critic", coeff=1.0)
        assert success, "activate() should return True"

        v_loaded = injector.get_normalized_vector()

        # Verify shape
        assert v_loaded.shape == (1, 1, d), \
            f"Expected shape [1,1,{d}], got {list(v_loaded.shape)}"

        # Verify normalization (should be unit vector)
        norm = v_loaded.view(-1).norm().item()
        assert abs(norm - 1.0) < 1e-5, \
            f"Expected unit norm, got {norm:.6f}"

        # Verify direction matches original
        v_orig_normalized = v_original / v_original.norm()
        cosine_sim = torch.nn.functional.cosine_similarity(
            v_loaded.view(1, -1), v_orig_normalized.view(1, -1)
        ).item()
        assert abs(cosine_sim - 1.0) < 1e-5, \
            f"Direction mismatch, cosine_sim={cosine_sim:.6f}"

        injector.deactivate()
        assert not injector.is_active, "Should be inactive after deactivate"

        print("  ✅ PASS — Save/Load cycle works correctly")
        print(f"    Shape: {list(v_loaded.shape)}, Norm: {norm:.6f}, Cosine: {cosine_sim:.6f}")
        return True
    finally:
        shutil.rmtree(tmpdir)


def test_raw_fallback():
    """Test: when only raw vector exists, VectorInjector falls back to it."""
    print("\n" + "="*60)
    print("  TEST 2: Raw Vector Fallback")
    print("="*60)

    tmpdir = tempfile.mkdtemp(prefix="test_vectors_")
    try:
        d = 4096
        v = torch.randn(d)

        # Only save as raw (no purified version)
        torch.save(v, os.path.join(tmpdir, "critic_raw.pt"))

        injector = VectorInjector(tmpdir, device="cpu", model_dtype=torch.float32)
        success = injector.activate("critic")
        assert success, "Should fall back to raw vector"

        v_loaded = injector.get_normalized_vector()
        assert v_loaded.shape == (1, 1, d)

        injector.deactivate()
        print("  ✅ PASS — Raw vector fallback works correctly")
        return True
    finally:
        shutil.rmtree(tmpdir)


def test_missing_vector():
    """Test: activate returns False when no vector file exists."""
    print("\n" + "="*60)
    print("  TEST 3: Missing Vector Handling")
    print("="*60)

    tmpdir = tempfile.mkdtemp(prefix="test_vectors_")
    try:
        injector = VectorInjector(tmpdir, device="cpu", model_dtype=torch.float32)
        success = injector.activate("nonexistent")
        assert not success, "Should return False for missing vector"

        print("  ✅ PASS — Missing vector handled correctly")
        return True
    finally:
        shutil.rmtree(tmpdir)


def test_manifold_projector():
    """Test: ManifoldProjector fit → purify → shape/correctness."""
    print("\n" + "="*60)
    print("  TEST 4: ManifoldProjector PCA Round-trip")
    print("="*60)

    d = 4096
    n_samples = 200
    n_components = 10

    # Generate fake activation matrix
    activations = np.random.randn(n_samples, d).astype(np.float32)

    # Create and fit projector
    projector = ManifoldProjector(n_components=n_components)
    projector.fit(activations)

    assert projector.components is not None, "Components should be set after fit"
    assert projector.components.shape == (n_components, d), \
        f"Expected shape ({n_components}, {d}), got {projector.components.shape}"

    # Purify a random vector
    v_raw = torch.randn(d)
    v_purified = projector.purify_vector(v_raw)

    assert v_purified.shape == v_raw.shape, \
        f"Shape mismatch: {v_purified.shape} vs {v_raw.shape}"

    # Verify purified vector lies in the PCA subspace
    # (projection of purified onto PCA subspace should equal itself)
    v_reprojected = projector.purify_vector(v_purified)
    diff = (v_purified - v_reprojected).norm().item()
    assert diff < 1e-3, f"Re-projection error too large: {diff:.6f}"

    print("  ✅ PASS — ManifoldProjector works correctly")
    print(f"    Components: {projector.components.shape}")
    print(f"    Explained variance: {projector.pca.explained_variance_ratio_.sum():.4f}")
    print(f"    Re-projection error: {diff:.6f}")
    return True


def test_manifold_save_load():
    """Test: ManifoldProjector save → load components."""
    print("\n" + "="*60)
    print("  TEST 5: ManifoldProjector Save/Load Components")
    print("="*60)

    tmpdir = tempfile.mkdtemp(prefix="test_pca_")
    try:
        d = 4096
        n_samples = 200
        n_components = 10

        activations = np.random.randn(n_samples, d).astype(np.float32)

        # Fit and save
        proj1 = ManifoldProjector(n_components=n_components)
        proj1.fit(activations)
        pca_path = os.path.join(tmpdir, "pca_components.npy")
        proj1.save_components(pca_path)

        # Load into a new projector
        proj2 = ManifoldProjector()
        proj2.load_components(pca_path)

        assert proj2.components is not None
        assert np.allclose(proj1.components, proj2.components, atol=1e-6), \
            "Loaded components should match saved components"

        # Verify both produce the same purification
        v_raw = torch.randn(d)
        v1 = proj1.purify_vector(v_raw)
        v2 = proj2.purify_vector(v_raw)
        diff = (v1 - v2).norm().item()
        assert diff < 1e-5, f"Purification mismatch: {diff:.6f}"

        print("  ✅ PASS — Save/Load components works correctly")
        print(f"    Difference between saved and loaded purification: {diff:.6f}")
        return True
    finally:
        shutil.rmtree(tmpdir)


def test_spherical_integration():
    """Test: loaded vector works with spherical_rotate, norm is preserved."""
    print("\n" + "="*60)
    print("  TEST 6: Integration with Spherical Steering")
    print("="*60)

    tmpdir = tempfile.mkdtemp(prefix="test_vectors_")
    try:
        d = 4096
        v = torch.randn(d)
        v = v / v.norm()  # pre-normalize
        torch.save(v, os.path.join(tmpdir, "critic.pt"))

        # Load via VectorInjector
        injector = VectorInjector(tmpdir, device="cpu", model_dtype=torch.float32)
        injector.activate("critic")
        v_loaded = injector.get_normalized_vector()  # [1, 1, d]

        # Create a fake hidden state batch
        h = torch.randn(1, 1, d) * 50.0  # arbitrary large norm
        h_norm_before = h.view(-1).norm().item()

        # Apply spherical rotation with alpha=0.15 radians
        alpha = 0.15
        h_new = spherical_rotate(h, v_loaded, alpha)

        h_norm_after = h_new.view(-1).norm().item()

        # Norm should be preserved
        norm_diff = abs(h_norm_after - h_norm_before)
        assert norm_diff < 1e-3, \
            f"Norm not preserved: before={h_norm_before:.4f}, after={h_norm_after:.4f}"

        # Output should differ from input (rotation happened)
        change = (h_new - h).norm().item()
        assert change > 1e-3, "Rotation should change the hidden state"

        injector.deactivate()
        print("  ✅ PASS — Spherical steering integration works correctly")
        print(f"    Norm before: {h_norm_before:.4f}")
        print(f"    Norm after:  {h_norm_after:.4f}")
        print(f"    Change:      {change:.4f}")
        return True
    finally:
        shutil.rmtree(tmpdir)


def test_full_pipeline_simulation():
    """Test: Simulate the full extraction → purify → save → load → steer cycle."""
    print("\n" + "="*60)
    print("  TEST 7: Full Pipeline Simulation")
    print("="*60)

    tmpdir = tempfile.mkdtemp(prefix="test_full_")
    try:
        d = 4096
        n_pairs = 50
        n_components = 10

        # Step 1: Simulate CAA extraction
        pos_acts = torch.randn(n_pairs, d)
        neg_acts = torch.randn(n_pairs, d)
        v_raw = pos_acts.mean(dim=0) - neg_acts.mean(dim=0)
        print(f"  Step 1: Simulated CAA extraction, v_raw norm = {v_raw.norm():.4f}")

        # Step 2: Save raw vector
        torch.save(v_raw, os.path.join(tmpdir, "critic_raw.pt"))

        # Step 3: PCA purification
        all_acts = torch.cat([pos_acts, neg_acts], dim=0).numpy()
        projector = ManifoldProjector(n_components=n_components)
        projector.fit(all_acts)

        v_purified = projector.purify_vector(v_raw)
        v_purified_flat = v_purified.view(-1)
        v_purified_flat = v_purified_flat / v_purified_flat.norm()
        v_purified = v_purified_flat.view(v_raw.shape)
        print(f"  Step 3: PCA purified, v_purified norm = {v_purified.view(-1).norm():.4f}")

        # Step 4: Save purified vector
        torch.save(v_purified, os.path.join(tmpdir, "critic.pt"))
        projector.save_components(os.path.join(tmpdir, "pca_components.npy"))

        # Step 5: Load via VectorInjector (as run_experiment.py would)
        injector = VectorInjector(tmpdir, device="cpu", model_dtype=torch.float32)
        success = injector.activate("critic", coeff=1.0)
        assert success
        v_for_steering = injector.get_normalized_vector()
        print(f"  Step 5: Loaded via VectorInjector, shape = {list(v_for_steering.shape)}")

        # Step 6: Use in spherical rotation
        h = torch.randn(1, 1, d) * 30.0
        h_norm = h.view(-1).norm().item()
        h_steered = spherical_rotate(h, v_for_steering, 0.1)
        h_steered_norm = h_steered.view(-1).norm().item()

        assert abs(h_steered_norm - h_norm) < 1e-2
        print(f"  Step 6: Spherical rotation OK, norm preserved: "
              f"{h_norm:.2f} → {h_steered_norm:.2f}")

        injector.deactivate()
        print("\n  ✅ PASS — Full pipeline simulation completed successfully!")
        return True
    finally:
        shutil.rmtree(tmpdir)


# ------------------------------------------------------------------ #
#  Main                                                               #
# ------------------------------------------------------------------ #

def main():
    print("="*60)
    print("  Vector Pipeline — End-to-End Test Suite")
    print("="*60)

    tests = [
        test_save_load_cycle,
        test_raw_fallback,
        test_missing_vector,
        test_manifold_projector,
        test_manifold_save_load,
        test_spherical_integration,
        test_full_pipeline_simulation,
    ]

    results = []
    for test_fn in tests:
        try:
            passed = test_fn()
            results.append((test_fn.__name__, passed))
        except Exception as e:
            print(f"  ❌ FAIL — {test_fn.__name__}: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_fn.__name__, False))

    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    all_pass = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  {name}")
        if not passed:
            all_pass = False

    print()
    if all_pass:
        print("  🎉 All tests passed! Pipeline is ready for GPU extraction.")
    else:
        print("  ⚠️  Some tests failed. Please check the errors above.")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
