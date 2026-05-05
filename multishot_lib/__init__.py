"""
Multishot analysis library for analysislib_v2

Provides utilities for:
- Loading and filtering multiple shots based on quality metrics and parameter consistency
- Checking that raw image analysis parameters are identical across shots
- Rejecting shots based on saturation, background, probe normalization errors, and DMD update status
- Generating statistics on rejection reasons
"""

from .main import get_day_data, get_and_filter_shots

__all__ = ['get_day_data', 'get_and_filter_shots']
