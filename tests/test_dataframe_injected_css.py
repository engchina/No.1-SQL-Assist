import unittest
from pathlib import Path


class DataframeInjectedCssTest(unittest.TestCase):
    def test_injected_css_does_not_collapse_borders(self):
        # Gradio Dataframe の仮想スクロールは tbody の padding で未描画行の高さを確保する。
        # border-collapse: collapse だと padding が無視され、途中の行までしかスクロールできない。
        utils_dir = Path(__file__).resolve().parents[1] / "utils"
        offenders = [
            path.name
            for path in sorted(utils_dir.glob("*.py"))
            if "border-collapse: collapse" in path.read_text(encoding="utf-8")
        ]
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
