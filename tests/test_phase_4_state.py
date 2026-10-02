"""
Unit tests for Phase 4 / Phase 6: Final ItaNet-111 State Representation.
"""
import os
import numpy as np
import pandas as pd
import pytest

from src.models.itanet import (
    IMAGE_DIM,
    TEXT_DIM,
    ATTRIBUTE_DIM,
    AVAILABLE_STATE_DIM,
    PAPER_STATE_DIM,
    ItaNetStateBuilder
)

def test_dimensions():
    assert IMAGE_DIM == 32
    assert TEXT_DIM == 32
    assert ATTRIBUTE_DIM == 47
    assert AVAILABLE_STATE_DIM == 111
    assert PAPER_STATE_DIM == 121

def test_state_files_exist():
    assert os.path.exists("data/processed/state/state.npy")
    assert os.path.exists("data/processed/state/state_row_indices.npy")
    assert os.path.exists("data/processed/state/state_train.npy")
    assert os.path.exists("data/processed/state/state_val.npy")
    assert os.path.exists("data/processed/state/state_test.npy")
    assert os.path.exists("data/processed/state/state_validity.csv")

def test_state_shape_and_finite():
    state = np.load("data/processed/state/state.npy")
    assert state.ndim == 2
    assert state.shape == (34028, 111)
    assert np.isfinite(state).all()

def test_state_row_indices():
    indices = np.load("data/processed/state/state_row_indices.npy")
    assert len(indices) == 34028
    assert indices.min() >= 0
    assert indices.max() < 34041
    assert np.all(np.diff(indices) > 0)  # strictly increasing

def test_state_splits():
    train = np.load("data/processed/state/state_train.npy")
    val = np.load("data/processed/state/state_val.npy")
    test = np.load("data/processed/state/state_test.npy")
    assert train.shape == (20179, 111)
    assert val.shape == (6512, 111)
    assert test.shape == (7337, 111)
    assert len(train) + len(val) + len(test) == 34028

def test_state_validity_records():
    df = pd.read_csv("data/processed/state/state_validity.csv")
    assert len(df) == 34041
    invalid = df[df["complete_state_valid"] == False]
    assert len(invalid) == 13
    assert set(invalid["invalid_reason"].unique()) == {"Missing Image Feature"}

def test_itanet_state_builder():
    img = np.random.randn(5, 32).astype(np.float32)
    txt = np.random.randn(5, 32).astype(np.float32)
    att = np.random.randn(5, 47).astype(np.float32)
    builder = ItaNetStateBuilder(image_features=img, text_features=txt, attribute_features=att)
    st = builder.build()
    assert st.shape == (5, 111)
    assert builder.state_dim == 111
