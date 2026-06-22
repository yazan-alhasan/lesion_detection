import unittest

import cv2
import numpy as np

from src.data.pectoral_suppression import (
    apply_pectoral_attenuation,
    detect_pectoral_mask,
    is_mlo_view,
    suppress_pectoral_muscle,
    to_uint8_gray,
)


class PectoralSuppressionTests(unittest.TestCase):
    @staticmethod
    def make_mlo(side: str) -> np.ndarray:
        image = np.full((400, 300), 40, dtype=np.uint8)
        if side == "left":
            points = np.array([[0, 0], [125, 0], [0, 220]], dtype=np.int32)
        else:
            points = np.array([[299, 0], [174, 0], [299, 220]], dtype=np.int32)
        cv2.fillPoly(image, [points], 240)
        return image

    def test_view_detection(self):
        self.assertTrue(is_mlo_view("LMLO"))
        self.assertTrue(is_mlo_view("rmlo"))
        self.assertFalse(is_mlo_view("LCC"))
        self.assertFalse(is_mlo_view(None))

    def test_suppresses_both_orientations_without_resizing(self):
        for side in ("left", "right"):
            with self.subTest(side=side):
                image = self.make_mlo(side)
                output, mask, info = suppress_pectoral_muscle(image, "MLO")
                self.assertTrue(info["applied"])
                self.assertEqual(info["side"], side)
                self.assertEqual(output.shape, image.shape)
                self.assertEqual(mask.shape, image.shape)
                self.assertGreater(np.count_nonzero(mask), 0)

    def test_cc_image_is_unchanged(self):
        image = self.make_mlo("left")
        output, mask, info = suppress_pectoral_muscle(image, "CC")
        np.testing.assert_array_equal(output, image)
        self.assertFalse(mask.any())
        self.assertEqual(info["reason"], "not_mlo_view")

    def test_uint16_rgb_conversion_preserves_dimensions(self):
        image = np.dstack([self.make_mlo("left")] * 3).astype(np.uint16) * 200
        gray = to_uint8_gray(image)
        self.assertEqual(gray.shape, image.shape[:2])
        self.assertEqual(gray.dtype, np.uint8)

    def test_rejects_unknown_mode(self):
        with self.assertRaises(ValueError):
            suppress_pectoral_muscle(self.make_mlo("left"), "MLO", suppress_mode="erase")

    def test_attenuation_factor_is_configurable(self):
        image = self.make_mlo("left")
        output, mask, info = suppress_pectoral_muscle(
            image, "MLO", suppress_mode="attenuate", attenuation_factor=0.5
        )
        self.assertTrue(info["applied"])
        np.testing.assert_array_equal(output[mask > 0], (image[mask > 0] * 0.5).astype(np.uint8))

    def test_default_mode_is_attenuation(self):
        image = self.make_mlo("left")
        output, mask, info = suppress_pectoral_muscle(image, "MLO")
        self.assertEqual(info["suppress_mode"], "attenuate")
        self.assertTrue(np.all(output[mask > 0] > 0))

    def test_unsafe_mask_area_is_skipped(self):
        image = self.make_mlo("left")
        for minimum, maximum, expected_reason in (
            (0.2, 0.3, "mask_too_small"),
            (0.001, 0.05, "mask_too_large"),
        ):
            with self.subTest(reason=expected_reason):
                mask, info = detect_pectoral_mask(
                    image, "MLO", min_area_ratio=minimum, max_area_ratio=maximum
                )
                self.assertFalse(info["applied"])
                self.assertEqual(info["reason"], expected_reason)
                self.assertFalse(mask.any())

    def test_attenuation_rejects_invalid_factor(self):
        with self.assertRaises(ValueError):
            apply_pectoral_attenuation(self.make_mlo("left"), np.ones((400, 300), np.uint8), 1.1)

    def test_mask_extending_too_low_is_skipped(self):
        image = self.make_mlo("left")
        mask, info = detect_pectoral_mask(
            image,
            "MLO",
            max_height_ratio=0.5,
            max_y_ratio=0.25,
        )
        self.assertFalse(info["applied"])
        self.assertEqual(info["reason"], "mask_extends_too_low")
        self.assertFalse(mask.any())


if __name__ == "__main__":
    unittest.main()
