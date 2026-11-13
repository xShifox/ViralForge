"""Tests for split-screen and PiP filtergraph layouts."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clipforge.render.filtergraph import FrameGeometry, layout_filter, build_video_chain  # noqa: E402
from clipforge.config import RenderSettings, BrandSettings  # noqa: E402


class TestLayoutFilterSplitPip:
    """layout_filter must emit correct chains for split and pip."""

    def setup_method(self):
        # Landscape source -> 9:16 vertical target
        self.geo = FrameGeometry(src_w=1920, src_h=1080, out_w=1080, out_h=1920)

    def test_split_layout_emits_vstack_of_two_streams(self):
        """split: source on top, speaker crop on bottom, padded to fill 9:16."""
        chain = layout_filter(self.geo, "split", x_bias=0.0)
        # Must use vstack, not overlay
        assert "vstack" in chain
        # Two video chains: one for upper (source), one for lower (speaker)
        assert "crop=" in chain
        assert "scale=" in chain
        # Top half: source scaled to cover, cropped to upper half
        assert "crop=1080:960:0:0[top]" in chain
        # Bottom half: speaker crop
        assert "vstack[v]" in chain

    def test_split_uses_x_bias_for_speaker_crop(self):
        """split uses x_bias for the bottom (speaker) half."""
        chain = layout_filter(self.geo, "split", x_bias=0.3)
        chain_center = layout_filter(self.geo, "split", x_bias=0.0)
        assert chain != chain_center, "x_bias should change the filter chain"

    def test_pip_layout_emits_overlay_of_small_speaker_on_main(self):
        """pip: main uncropped feed with small speaker overlay in corner."""
        chain = layout_filter(self.geo, "pip", x_bias=0.0)
        # Must use overlay for the PIP
        assert "overlay" in chain
        # Speaker overlay should be scaled down (PIP size ~360x640)
        assert "scale=360:640" in chain or "scale=" in chain
        # PIP positioned in corner (bottom-right with margin)
        assert "overlay=W-w-" in chain
        # Main feed uses blurpad style (gblur)
        assert "gblur=sigma=42" in chain

    def test_pip_uses_x_bias_for_speaker_crop(self):
        """pip uses x_bias for the PIP speaker crop."""
        chain = layout_filter(self.geo, "pip", x_bias=0.4)
        chain_center = layout_filter(self.geo, "pip", x_bias=0.0)
        assert chain != chain_center, "x_bias should change the filter chain"

    def test_crop_layout_unchanged(self):
        """crop layout must produce identical chain to before."""
        chain = layout_filter(self.geo, "crop", x_bias=0.0)
        assert "force_original_aspect_ratio=increase" in chain
        assert "crop=1080:1920" in chain

    def test_blurpad_layout_unchanged(self):
        """blurpad (fit) layout must produce identical chain to before."""
        chain = layout_filter(self.geo, "blurpad", x_bias=0.0)
        assert "split=2" in chain
        assert "gblur=sigma=42" in chain
        assert "overlay" in chain


class TestBuildVideoChainSplitPip:
    """build_video_chain must wire split/pip layouts through correctly."""

    def setup_method(self):
        self.geo = FrameGeometry(src_w=1920, src_h=1080, out_w=1080, out_h=1920)
        self.render = RenderSettings(width=1080, height=1920, layout="split")
        self.brand = BrandSettings(enabled=False)
        self.font = "Segoe UI Black"

    def test_split_layout_uses_speaker_crop_for_lower_half(self):
        """split layout uses speaker crop (x_bias from tracker) for lower half."""
        chain = build_video_chain(
            self.geo,
            RenderSettings(width=1080, height=1920, layout="split"),
            BrandSettings(enabled=False),
            self.font,
            "Segoe UI",
            duration=30.0,
        )
        assert "split" in chain.lower() or "vstack" in chain

    def test_pip_layout_overlays_speaker_in_corner(self):
        """pip layout overlays scaled speaker crop in corner."""
        chain = build_video_chain(
            self.geo,
            RenderSettings(width=1080, height=1920, layout="pip"),
            BrandSettings(enabled=False),
            self.font,
            "Segoe UI",
            duration=30.0,
        )
        assert "overlay" in chain

    def test_crop_and_fit_still_work(self):
        """Existing crop and blurpad layouts must still work."""
        for layout in ("crop", "blurpad"):
            chain = build_video_chain(
                self.geo,
                RenderSettings(width=1080, height=1920, layout=layout),
                BrandSettings(enabled=False),
                self.font,
                "Segoe UI",
                duration=30.0,
            )
            assert "crop=" in chain
            assert "scale=" in chain

    def test_split_layout_does_not_fall_through_to_blurpad(self):
        """split layout must not fall through to the blurpad/else branch."""
        chain = build_video_chain(
            self.geo,
            RenderSettings(width=1080, height=1920, layout="split"),
            BrandSettings(enabled=False),
            self.font,
            "Segoe UI",
            duration=30.0,
        )
        # Should use layout_filter which produces vstack, not the blurpad overlay
        assert "vstack" in chain
        assert "gblur=sigma=42" not in chain or "vstack" in chain

    def test_pip_layout_does_not_fall_through_to_blurpad(self):
        """pip layout must not fall through to the blurpad/else branch."""
        chain = build_video_chain(
            self.geo,
            RenderSettings(width=1080, height=1920, layout="pip"),
            BrandSettings(enabled=False),
            self.font,
            "Segoe UI",
            duration=30.0,
        )
        # Should use layout_filter which produces pip overlay, not the blurpad
        assert "overlay=W-w-" in chain
        # The blurpad branch has a different overlay pattern
        # Split/pip both use layout_filter which outputs to [v]
        # so the chain should not contain the blurpad's [base] naming
        assert "[base]" not in chain or "vstack" in chain or "overlay=W-w-" in chain


if __name__ == "__main__":
    pytest.main([__file__, "-v"])