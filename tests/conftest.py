"""Shared pytest fixtures."""

from __future__ import annotations

import warnings

import pytest
from pymatgen.core import Lattice, Structure

from matscout.config.loader import load_campaign_config, load_global_config
from matscout.data.mp_client import fetch_mp_seeds

warnings.simplefilter("ignore")


@pytest.fixture(scope="session")
def global_cfg():
    return load_global_config()


@pytest.fixture
def battery_cfg():
    return load_campaign_config("battery")


@pytest.fixture
def seeds(battery_cfg, global_cfg):
    return fetch_mp_seeds(battery_cfg, global_cfg, limit=50)


@pytest.fixture
def nacl_structure():
    return Structure(Lattice.cubic(5.64), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])


@pytest.fixture
def overlapping_structure():
    # two atoms essentially on top of each other
    return Structure(Lattice.cubic(5.0), ["Na", "Cl"], [[0, 0, 0], [0.02, 0, 0]])
