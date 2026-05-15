import os
import unittest

from ppe_model_registry import (
    PPE_MODEL_PROFILES,
    build_capability_summary,
    get_profile,
    normalize_class_name,
)


class TestPPEModelRegistry(unittest.TestCase):
    def test_hexmon_profile_exists(self):
        profile = get_profile("hexmon")
        self.assertEqual(profile.repo_id, "Hexmon/vyra-yolo-ppe-detection")
        self.assertEqual(profile.filename, "best.pt")
        self.assertEqual(profile.local_path, os.path.join("models", "hexmon_vyra_yolo_ppe_best.pt"))
        self.assertIn("NO-Hardhat", profile.expected_classes)

    def test_hansung_profile_exists(self):
        profile = get_profile("hansung")
        self.assertEqual(profile.repo_id, "Hansung-Cho/yolov8-ppe-detection")
        self.assertEqual(profile.filename, "best.pt")
        self.assertEqual(profile.local_path, os.path.join("models", "hansung_yolov8_ppe_best.pt"))
        self.assertIn("No-Mask", profile.expected_classes)

    def test_registry_contains_both_profiles(self):
        self.assertIn("hexmon_vyra_yolo_ppe", PPE_MODEL_PROFILES)
        self.assertIn("hansung_yolov8_ppe", PPE_MODEL_PROFILES)

    def test_class_normalization(self):
        self.assertEqual(normalize_class_name("NO-Hardhat"), "no_helmet")
        self.assertEqual(normalize_class_name("no_hardhat"), "no_helmet")
        self.assertEqual(normalize_class_name("No-Safety Vest"), "no_safety_vest")
        self.assertEqual(normalize_class_name("Safety Vest"), "safety_vest")
        self.assertEqual(normalize_class_name("Person"), "person")

    def test_hexmon_capabilities(self):
        caps = build_capability_summary(get_profile("hexmon").expected_classes)
        self.assertTrue(caps["person"])
        self.assertTrue(caps["helmet"])
        self.assertTrue(caps["no_helmet"])
        self.assertTrue(caps["safety_vest"])
        self.assertTrue(caps["no_safety_vest"])

    def test_hansung_capabilities(self):
        caps = build_capability_summary(get_profile("hansung").expected_classes)
        self.assertTrue(caps["person"])
        self.assertTrue(caps["helmet"])
        self.assertTrue(caps["no_helmet"])
        self.assertTrue(caps["safety_vest"])
        self.assertTrue(caps["no_safety_vest"])
        self.assertTrue(caps["mask"])
        self.assertTrue(caps["no_mask"])


if __name__ == "__main__":
    unittest.main()
