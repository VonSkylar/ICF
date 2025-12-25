"""
STL file loading and mesh processing utilities.
"""

from __future__ import annotations

import os
import struct
from typing import List, Optional

import numpy as np


def load_stl_mesh(file_path: str) -> np.ndarray:
    """Load a mesh from an STL file.

    The STL format can be either ASCII or binary. This routine attempts to
    detect the format automatically and returns an array of facets, where each
    facet is represented by three 3‑D vertices. The ordering of the vertices
    follows the order in the file and is not otherwise interpreted.

    Parameters
    ----------
    file_path : str
        Path to the STL file on disk.

    Returns
    -------
    np.ndarray, shape (n_facets, 4, 3)
        An array containing all triangle data. Each facet is of shape
        (4, 3) and contains [normal, v0, v1, v2].
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"STL file '{file_path}' does not exist")

    file_size = os.path.getsize(file_path)
    with open(file_path, 'rb') as f:
        header = f.read(80)
        count_bytes = f.read(4)
        triangle_count = int.from_bytes(count_bytes, byteorder='little', signed=False) if len(count_bytes) == 4 else 0
    header_str = header.decode(errors='ignore').strip().lower()
    expected_binary_size = 84 + triangle_count * 50 if triangle_count > 0 else None

    def _try_load_binary() -> Optional[np.ndarray]:
        if expected_binary_size is None or expected_binary_size > file_size:
            return None
        if expected_binary_size != file_size:
            return None
        facets_bin: List[np.ndarray] = []
        record_struct = struct.Struct('<12fH')
        with open(file_path, 'rb') as bf:
            bf.seek(84)
            for _ in range(triangle_count):
                chunk = bf.read(record_struct.size)
                if len(chunk) != record_struct.size:
                    return None
                unpacked = record_struct.unpack(chunk)
                normal = np.array(unpacked[0:3], dtype=float)
                v0 = np.array(unpacked[3:6], dtype=float)
                v1 = np.array(unpacked[6:9], dtype=float)
                v2 = np.array(unpacked[9:12], dtype=float)
                facets_bin.append(np.stack([normal, v0, v1, v2], axis=0))
        if not facets_bin:
            return None
        return np.stack(facets_bin, axis=0)

    def _try_load_ascii() -> Optional[np.ndarray]:
        facets_ascii: List[np.ndarray] = []
        current_vertices: List[np.ndarray] = []
        current_normal: Optional[np.ndarray] = None
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as af:
            for line in af:
                tokens = line.strip().split()
                if not tokens:
                    continue
                keyword = tokens[0].lower()
                if keyword == 'facet' and len(tokens) >= 5 and tokens[1].lower() == 'normal':
                    current_normal = np.array(list(map(float, tokens[2:5])), dtype=float)
                    current_vertices = []
                elif keyword == 'vertex' and len(tokens) >= 4:
                    v = np.array(list(map(float, tokens[1:4])), dtype=float)
                    current_vertices.append(v)
                elif keyword == 'endfacet':
                    if len(current_vertices) >= 3:
                        if current_normal is None:
                            v0, v1, v2 = current_vertices[0], current_vertices[1], current_vertices[2]
                            edge1 = v1 - v0
                            edge2 = v2 - v0
                            current_normal = np.cross(edge1, edge2)
                            norm = np.linalg.norm(current_normal)
                            if norm > 0:
                                current_normal = current_normal / norm
                            else:
                                current_normal = np.array([0.0, 0.0, 1.0])
                        facets_ascii.append(np.stack([current_normal] + current_vertices[:3], axis=0))
                    current_normal = None
                    current_vertices = []
        if not facets_ascii:
            return None
        return np.stack(facets_ascii, axis=0)

    facets = None
    binary_first = True
    if header_str.startswith('solid') and (expected_binary_size is None or expected_binary_size != file_size):
        binary_first = False

    loaders = (_try_load_binary, _try_load_ascii) if binary_first else (_try_load_ascii, _try_load_binary)
    for loader in loaders:
        facets = loader()
        if facets is not None:
            break

    if facets is None:
        raise ValueError(f"No facets were found in '{file_path}' - the file may be corrupt")
    return facets
