import unittest

from utils.object_selector_util import (
    effective_object_selector_selection,
    filter_object_selector_choices,
    merge_object_selector_selection,
    normalize_object_selector_choices,
    visible_object_selector_value,
)


class SelectAiObjectSelectorTest(unittest.TestCase):
    def test_normalize_choices_removes_empty_values_and_duplicates(self):
        self.assertEqual(
            normalize_object_selector_choices(
                [" EMPLOYEE ", "", None, "DEPARTMENT", "EMPLOYEE"]
            ),
            ["EMPLOYEE", "DEPARTMENT"],
        )

    def test_filter_choices_matches_case_insensitive_substrings(self):
        self.assertEqual(
            filter_object_selector_choices(
                ["EMPLOYEE", "DEPARTMENT", "V_EMP_DEPT"],
                "emp",
            ),
            ["EMPLOYEE", "V_EMP_DEPT"],
        )

    def test_filter_choices_returns_all_for_blank_query(self):
        self.assertEqual(
            filter_object_selector_choices(["EMPLOYEE", "DEPARTMENT"], " "),
            ["EMPLOYEE", "DEPARTMENT"],
        )

    def test_filter_choices_returns_empty_for_no_results(self):
        self.assertEqual(
            filter_object_selector_choices(["EMPLOYEE", "DEPARTMENT"], "invoice"),
            [],
        )

    def test_visible_value_keeps_only_selected_visible_items(self):
        self.assertEqual(
            visible_object_selector_value(
                ["EMPLOYEE", "DEPARTMENT"],
                ["DEPARTMENT", "HIDDEN_TABLE"],
            ),
            ["DEPARTMENT"],
        )

    def test_merge_selection_preserves_hidden_selected_items(self):
        self.assertEqual(
            merge_object_selector_selection(
                ["CUSTOMERS", "V_CUSTOMERS"],
                ["CUSTOMERS", "ORDERS"],
                ["ORDERS"],
            ),
            ["V_CUSTOMERS", "ORDERS"],
        )

    def test_effective_selection_merges_current_visible_value_for_submit(self):
        self.assertEqual(
            effective_object_selector_selection(
                ["ORDERS"],
                ["CUSTOMERS", "ORDERS"],
                ["CUSTOMERS", "V_CUSTOMERS"],
            ),
            ["V_CUSTOMERS", "ORDERS"],
        )

    def test_refresh_reset_has_no_selected_visible_values(self):
        self.assertEqual(
            visible_object_selector_value(["EMPLOYEE", "DEPARTMENT"], []),
            [],
        )


if __name__ == "__main__":
    unittest.main()
