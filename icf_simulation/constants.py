"""
Physical constants and simulation configuration.
"""

# Physical constants
AVOGADRO_CONSTANT = 6.02214076e23  # mol⁻¹
BARN_TO_M2 = 1.0e-28  # 1 barn = 10⁻²⁸ m²
NEUTRON_MASS_KG = 1.67492749804e-27  # kg

# Debug flag
DEBUG = False

# Global statistics for geometry leak monitoring
GEOMETRY_LEAK_STATS = {
    'total_queries': 0,
    'retry_success': 0,
    'retry_failures': 0,
    'outside_detections': 0,
}

# Default source cone half-angle (degrees)
DEFAULT_SOURCE_CONE_HALF_ANGLE_DEG = 10


def reset_geometry_leak_stats():
    """Reset geometry leak statistics counters."""
    global GEOMETRY_LEAK_STATS
    GEOMETRY_LEAK_STATS = {
        'total_queries': 0,
        'retry_success': 0,
        'retry_failures': 0,
        'outside_detections': 0,
    }


def print_geometry_leak_stats():
    """Print statistics about geometry leak handling.
    
    This helps diagnose mesh quality issues. High retry rates or failures
    indicate problems with the STL mesh (gaps, degenerate triangles, etc.)
    """
    stats = GEOMETRY_LEAK_STATS
    total = stats['total_queries']
    
    if total == 0:
        print("No geometry queries recorded.")
        return
    
    print("\n" + "="*60)
    print("GEOMETRY LEAK PREVENTION STATISTICS")
    print("="*60)
    print(f"Total exit queries:        {total:,}")
    print(f"Successful (first try):    {total - stats['retry_success'] - stats['retry_failures']:,} "
          f"({100*(total - stats['retry_success'] - stats['retry_failures'])/total:.2f}%)")
    print(f"Successful after retry:    {stats['retry_success']:,} "
          f"({100*stats['retry_success']/total:.2f}%)")
    print(f"Outside mesh detected:     {stats['outside_detections']:,} "
          f"({100*stats['outside_detections']/total:.2f}%)")
    print(f"True geometry leaks:       {stats['retry_failures']:,} "
          f"({100*stats['retry_failures']/total:.2f}%)")
    print("="*60)
    
    if stats['retry_failures'] > 0.01 * total:
        print("⚠️  WARNING: High geometry leak rate (>1%)")
        print("   This indicates poor mesh quality:")
        print("   - Check for gaps/holes in STL mesh")
        print("   - Look for degenerate triangles")
        print("   - Consider remeshing with proper tools")
    elif stats['retry_success'] > 0.05 * total:
        print("ℹ️  INFO: Moderate retry rate (>5%)")
        print("   Mesh has some numerical precision issues but is usable")
    else:
        print("✓ Mesh quality appears good")
    print()
