"""
Imaging Analysis Library v2

Refactored version with:
- Cleaner code organization and separation of concerns
- Data classes for type-safe data structures
- Comprehensive logging
- Processing parameter tracking for reproducibility
- Eliminated DMD-related code
"""

from .main import (
    # Constants
    ATOMIC_CROSS_SECTION_SIGMA,
    BACKGROUND_STD_MULTIPLIER_DEFAULT,
    SATURATION_THRESHOLD_MULTIPLIER_DEFAULT,
    F_TEST_THRESHOLD,
    PLOT_FONTSIZE,
    
    # Data classes
    ProcessedImageMetadata,
    ProcessedImages,
    ROIEdges,
    AxisInfo,
    
    # Core functions
    calculate_OD,
    get_n1Ds,
    get_axes,
    get_axes_infos,
    get_roiback_edges,
    get_roi_integration_edges,
    get_raws_camera,
    get_images_camera_waxes,
    get_images_camera_waxes_SVD,
    plot_OD,
    do_fit_1D,
    do_fit_2D,
    get_processing_params,
)

__all__ = [
    'ATOMIC_CROSS_SECTION_SIGMA',
    'BACKGROUND_STD_MULTIPLIER_DEFAULT',
    'SATURATION_THRESHOLD_MULTIPLIER_DEFAULT',
    'F_TEST_THRESHOLD',
    'PLOT_FONTSIZE',
    'ProcessedImageMetadata',
    'ProcessedImages',
    'ROIEdges',
    'AxisInfo',
    'calculate_OD',
    'get_n1Ds',
    'get_axes',
    'get_axes_infos',
    'get_roiback_edges',
    'get_roi_integration_edges',
    'get_raws_camera',
    'get_images_camera_waxes',
    'get_images_camera_waxes_SVD',
    'plot_OD',
    'do_fit_1D',
    'do_fit_2D',
    'get_processing_params',
]
