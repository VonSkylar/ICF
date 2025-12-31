"""
Validation utilities for testing geometry and simulation components.

This module provides functions to validate that the simulation components
are working correctly before running full simulations.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .simple_geometry import (
    create_simple_sphere,
    create_simple_tube,
    create_simple_box,
    create_simple_icf_geometry,
    print_mesh_info,
)


def validate_mesh(mesh: np.ndarray, name: str = "Mesh") -> Tuple[bool, str]:
    """Validate mesh data for correctness.
    
    Parameters
    ----------
    mesh : np.ndarray
        Mesh data to validate.
    name : str
        Name for error messages.
        
    Returns
    -------
    valid : bool
        True if mesh is valid.
    message : str
        Description of validation result.
    """
    errors = []
    
    # Check shape
    if mesh.ndim != 3:
        errors.append(f"Expected 3D array, got {mesh.ndim}D")
    elif mesh.shape[1] not in (3, 4):
        errors.append(f"Expected shape (n, 3, 3) or (n, 4, 3), got {mesh.shape}")
    elif mesh.shape[2] != 3:
        errors.append(f"Expected 3D coordinates, got {mesh.shape[2]}D")
    
    if errors:
        return False, f"{name}: " + "; ".join(errors)
    
    # Check for NaN or Inf
    if np.any(np.isnan(mesh)):
        errors.append("Contains NaN values")
    if np.any(np.isinf(mesh)):
        errors.append("Contains Inf values")
    
    # Check normals if present
    if mesh.shape[1] == 4:
        normals = mesh[:, 0, :]
        norms = np.linalg.norm(normals, axis=1)
        
        # Allow small tolerance for unit normals
        if not np.allclose(norms, 1.0, atol=1e-5):
            errors.append("Some normals are not unit vectors")
    
    if errors:
        return False, f"{name}: " + "; ".join(errors)
    
    return True, f"{name}: Valid ({mesh.shape[0]} facets)"


def validate_geometry_module() -> Tuple[bool, list]:
    """Validate that the simple geometry module works correctly.
    
    Returns
    -------
    success : bool
        True if all tests pass.
    results : list
        List of test results.
    """
    results = []
    all_passed = True
    
    # Test sphere creation
    try:
        mesh = create_simple_sphere(center=(0, 0, 0), radius=50.0, subdivisions=2)
        valid, msg = validate_mesh(mesh, "Sphere")
        results.append(("Sphere creation", valid, msg))
        if not valid:
            all_passed = False
    except Exception as e:
        results.append(("Sphere creation", False, str(e)))
        all_passed = False
    
    # Test tube creation
    try:
        mesh = create_simple_tube(
            start_point=(0, 0, 0),
            end_point=(0, 0, 3000),
            inner_radius=100.0,
            outer_radius=110.0,
            n_segments=16
        )
        valid, msg = validate_mesh(mesh, "Tube")
        results.append(("Tube creation", valid, msg))
        if not valid:
            all_passed = False
    except Exception as e:
        results.append(("Tube creation", False, str(e)))
        all_passed = False
    
    # Test box creation
    try:
        mesh = create_simple_box(center=(0, 0, 100), size=(200, 200, 200))
        valid, msg = validate_mesh(mesh, "Box")
        results.append(("Box creation", valid, msg))
        if not valid:
            all_passed = False
        
        # Box should have exactly 12 faces
        if mesh.shape[0] != 12:
            results.append(("Box face count", False, f"Expected 12 faces, got {mesh.shape[0]}"))
            all_passed = False
        else:
            results.append(("Box face count", True, "12 faces"))
    except Exception as e:
        results.append(("Box creation", False, str(e)))
        all_passed = False
    
    # Test ICF geometry creation
    for geom_type in ['minimal', 'standard', 'detailed']:
        try:
            shell, channel = create_simple_icf_geometry(geom_type)
            
            valid_shell, msg_shell = validate_mesh(shell, f"ICF Shell ({geom_type})")
            valid_channel, msg_channel = validate_mesh(channel, f"ICF Channel ({geom_type})")
            
            results.append((f"ICF {geom_type} shell", valid_shell, msg_shell))
            results.append((f"ICF {geom_type} channel", valid_channel, msg_channel))
            
            if not valid_shell or not valid_channel:
                all_passed = False
        except Exception as e:
            results.append((f"ICF {geom_type}", False, str(e)))
            all_passed = False
    
    return all_passed, results


def run_quick_test(verbose: bool = True) -> bool:
    """Run a quick validation test and print results.
    
    Parameters
    ----------
    verbose : bool
        Whether to print detailed results.
        
    Returns
    -------
    success : bool
        True if all tests pass.
        
    Example
    -------
    >>> from icf_simulation.testing import run_quick_test
    >>> success = run_quick_test()
    """
    if verbose:
        print("=" * 70)
        print("SIMPLE GEOMETRY MODULE VALIDATION")
        print("=" * 70)
    
    success, results = validate_geometry_module()
    
    if verbose:
        for test_name, passed, message in results:
            status = "✓" if passed else "✗"
            print(f"{status} {test_name}: {message}")
        
        print()
        print("=" * 70)
        if success:
            print("ALL TESTS PASSED ✓")
        else:
            print("SOME TESTS FAILED ✗")
        print("=" * 70)
    
    return success
