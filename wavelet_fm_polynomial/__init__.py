"""Controlled polynomial wavelet regressions for standard Flow Matching."""

from wavelet_fm_polynomial.features import (
    FeatureFamily,
    WaveletFeatureStructure,
    build_feature_structure,
    evaluate_features,
    feature_count,
)
from wavelet_fm_polynomial.regression import (
    PolynomialVelocityRegressor,
    fit_nested_degrees,
)
from wavelet_fm_polynomial.wiener import AffineFMWiener

__all__ = [
    "AffineFMWiener",
    "FeatureFamily",
    "PolynomialVelocityRegressor",
    "WaveletFeatureStructure",
    "build_feature_structure",
    "evaluate_features",
    "feature_count",
    "fit_nested_degrees",
]
