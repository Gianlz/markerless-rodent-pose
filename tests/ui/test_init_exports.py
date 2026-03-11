"""Tests for src.ui.styles and src.ui.widgets public API (re-exports)."""

import pytest


# ===================================================================
# styles __init__ re-exports
# ===================================================================


class TestStylesInit:
    """Ensure the styles package exposes the correct public API."""

    def test_load_stylesheet_importable(self) -> None:
        from src.ui.styles import load_stylesheet
        assert callable(load_stylesheet)

    def test_all_constants_importable(self) -> None:
        from src.ui.styles import (
            SECONDARY_BUTTON,
            DANGER_BUTTON,
            SUCCESS_LABEL,
            ERROR_LABEL,
            INFO_LABEL,
            VIDEO_LIST_LABEL,
        )

        for name in [
            SECONDARY_BUTTON,
            DANGER_BUTTON,
            SUCCESS_LABEL,
            ERROR_LABEL,
            INFO_LABEL,
            VIDEO_LIST_LABEL,
        ]:
            assert isinstance(name, str)

    def test_all_list_matches(self) -> None:
        from src.ui.styles import __all__

        expected = {
            "load_stylesheet",
            "SECONDARY_BUTTON",
            "DANGER_BUTTON",
            "SUCCESS_LABEL",
            "ERROR_LABEL",
            "INFO_LABEL",
            "VIDEO_LIST_LABEL",
        }
        assert set(__all__) == expected


# ===================================================================
# widgets __init__ re-exports
# ===================================================================


class TestWidgetsInit:
    """Ensure the widgets package exposes ResponsiveTabPage."""

    def test_responsive_tab_page_importable(self) -> None:
        from src.ui.widgets import ResponsiveTabPage
        assert ResponsiveTabPage is not None

    def test_all_list(self) -> None:
        from src.ui.widgets import __all__

        assert "ResponsiveTabPage" in __all__
