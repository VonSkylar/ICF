"""
Testing subpackage for ICF neutron simulation.

This subpackage provides tools for testing and debugging the simulation:
- Simple analytical geometry for isolating geometry-related issues
- Comparison utilities for STL vs simple geometry
- Validation functions

Example usage:
    from icf_simulation.testing import create_simple_icf_geometry, compare_geometries
    
    # Create simple geometry for testing
    shell_mesh, channel_mesh = create_simple_icf_geometry('standard')
    
    # Run comparison test
    results = compare_geometries(n_neutrons=100)
"""

from .simple_geometry import (
    create_simple_sphere,
    create_simple_tube,
    create_simple_box,
    create_simple_icf_geometry,
    print_mesh_info,
)

from .validation import (
    validate_mesh,
    validate_geometry_module,
    run_quick_test,
)

from .comparison import (
    compare_geometries,
    load_statistics,
    print_comparison,
)

__all__ = [
    # Simple geometry
    "create_simple_sphere",
    "create_simple_tube",
    "create_simple_box",
    "create_simple_icf_geometry",
    "print_mesh_info",
    # Validation
    "validate_mesh",
    "validate_geometry_module",
    "run_quick_test",
    # Comparison
    "compare_geometries",
    "load_statistics",
    "print_comparison",
]
