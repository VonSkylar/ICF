"""
Package data path utilities.

This module provides functions to access package-bundled data files
(cross-section data, STL models, etc.) regardless of where the package
is installed.

Example usage:
    from icf_simulation.data_paths import get_cross_section_dir, get_stl_dir
    
    al_csv = get_cross_section_dir() / "Al.csv"
    shell_stl = get_stl_dir() / "Target_ball_model.STL"
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import importlib.resources


def get_package_dir() -> Path:
    """Get the root directory of the icf_simulation package.
    
    Returns
    -------
    Path
        Path to the icf_simulation package directory.
    """
    return Path(__file__).resolve().parent


def get_data_dir() -> Path:
    """Get the path to the package data directory.
    
    Returns
    -------
    Path
        Path to icf_simulation/data/
    """
    return get_package_dir() / "data"


def get_cross_section_dir() -> Path:
    """Get the path to the cross-section data directory.
    
    Returns
    -------
    Path
        Path to icf_simulation/data/cross_sections/
        
    Example
    -------
    >>> from icf_simulation.data_paths import get_cross_section_dir
    >>> al_csv = get_cross_section_dir() / "Al.csv"
    """
    return get_data_dir() / "cross_sections"


def get_stl_dir() -> Path:
    """Get the path to the STL models directory.
    
    Returns
    -------
    Path
        Path to icf_simulation/data/stl_models/
        
    Example
    -------
    >>> from icf_simulation.data_paths import get_stl_dir
    >>> shell_stl = get_stl_dir() / "Target_ball_model.STL"
    """
    return get_data_dir() / "stl_models"


def get_examples_dir() -> Path:
    """Get the path to the examples directory.
    
    Returns
    -------
    Path
        Path to icf_simulation/examples/
    """
    return get_package_dir() / "examples"


def get_cross_section_file(element: str) -> Path:
    """Get the path to a specific cross-section data file.
    
    Parameters
    ----------
    element : str
        Element symbol (e.g., 'Al', 'H', 'C')
        
    Returns
    -------
    Path
        Path to the cross-section CSV file.
        
    Raises
    ------
    FileNotFoundError
        If the cross-section file does not exist.
    """
    file_path = get_cross_section_dir() / f"{element}.csv"
    if not file_path.exists():
        raise FileNotFoundError(
            f"Cross-section file for '{element}' not found at {file_path}. "
            f"Available files: {list(get_cross_section_dir().glob('*.csv'))}"
        )
    return file_path


def get_stl_file(name: str) -> Path:
    """Get the path to a specific STL model file.
    
    Parameters
    ----------
    name : str
        STL file name (with or without .STL extension)
        
    Returns
    -------
    Path
        Path to the STL file.
        
    Raises
    ------
    FileNotFoundError
        If the STL file does not exist.
    """
    if not name.upper().endswith('.STL'):
        name = f"{name}.STL"
    
    file_path = get_stl_dir() / name
    if not file_path.exists():
        # Try case-insensitive search
        for f in get_stl_dir().glob('*.[Ss][Tt][Ll]'):
            if f.name.lower() == name.lower():
                return f
        raise FileNotFoundError(
            f"STL file '{name}' not found at {file_path}. "
            f"Available files: {list(get_stl_dir().glob('*.[Ss][Tt][Ll]'))}"
        )
    return file_path


def list_cross_section_files() -> list[Path]:
    """List all available cross-section data files.
    
    Returns
    -------
    list[Path]
        List of paths to cross-section CSV files.
    """
    return list(get_cross_section_dir().glob('*.csv'))


def list_stl_files() -> list[Path]:
    """List all available STL model files.
    
    Returns
    -------
    list[Path]
        List of paths to STL files.
    """
    return list(get_stl_dir().glob('*.[Ss][Tt][Ll]'))
