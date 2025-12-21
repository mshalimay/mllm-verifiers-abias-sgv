import unittest
from typing import Union

from playwright.sync_api import Page, sync_playwright

# This is a placeholder for the type hint in the function signature.
# In a real scenario, you might have a more complex setup.
PseudoPage = object


def shopping_get_product_price(page: Page | PseudoPage) -> Union[float, int]:
    """Get the price of the product on the shopping website."""
    try:
        result = page.evaluate(
            """
                (() => {
                    try {
                        // get the price from the product page
                        const el = document.querySelector("#maincontent > div.columns > div > div.product-info-main > div.product-info-price > div.price-box.price-final_price > span > span");

                        if (!el) { return 0; }
                        const raw = el.outerText.trim();

                        // replace all non-numeric characters with an empty string
                        const s = raw.replace(/[^\d.,]/g, "");

                        let normalized = s;
                        const hasComma = s.includes(',');
                        const hasPeriod = s.includes('.');

                        // If both commas and periods are present, assume the rightmost is the decimal separator.
                        if (hasComma && hasPeriod) {
                            if (s.lastIndexOf(',') > s.lastIndexOf('.')) {
                                normalized = s.replace(/\./g, '').replace(',', '.');
                            } else {
                                normalized = s.replace(/,/g, '');
                            }
                            
                        // If only commas are present
                        } else if (hasComma && !hasPeriod) {
                            // Always treat "," as a thousands separator.
                            normalized = s.replace(/,/g, '');
                            // Potentially add some heuristic to handle "," as a decimal separator


                        // If only periods are present
                        } else if (hasPeriod && !hasComma) {
                            // If there are multiple periods, they must be thousands separators.
                            if ((s.match(/\./g) || []).length > 1) {
                                normalized = s.replace(/\./g, '');
                            }
                            // Otherwise, it's a single period; assume it's a decimal separator
                            // Potentially add some heuristic to handle "." as a thousands separator
                        }
                        // Else, no comma or period ==> no action is needed
                        
                        const n = parseFloat(normalized);
                        return Number.isFinite(n) ? n : 0;
                    } catch (e) {
                        return 0;
                    }
                })();
            """
        )
    except Exception:
        result = 0

    return result


# TEST SUITE
class TestShoppingPriceParser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up a browser instance once for all tests."""
        cls.playwright = sync_playwright().start()
        cls.browser = cls.playwright.chromium.launch()

    @classmethod
    def tearDownClass(cls):
        """Tear down the browser instance after all tests are done."""
        cls.browser.close()
        cls.playwright.stop()

    def setUp(self):
        """Create a new page for each individual test."""
        self.page = self.browser.new_page()

    def tearDown(self):
        """Close the page after each test."""
        self.page.close()

    def _run_test_with_price_string(self, price_string: str, expected_value: float):
        """Helper method to set content and run the parser."""
        html_content = f"""
            <div id="maincontent">
                <div class="columns">
                    <div>
                        <div class="product-info-main">
                            <div class="product-info-price">
                                <div class="price-box price-final_price">
                                    <span>
                                        <span>{price_string}</span>
                                    </span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        """
        self.page.set_content(html_content)
        result = shopping_get_product_price(self.page)
        self.assertAlmostEqual(result, expected_value, places=2, msg=f"Failed on input: '{price_string}'")

    def test_mixed_separators(self):
        """Tests prices with both period and comma separators."""
        self._run_test_with_price_string("$1,196.98", 1196.98)  # US format
        self._run_test_with_price_string("€1.196,98", 1196.98)  # European format
        self._run_test_with_price_string("1,234,567.89", 1234567.89)
        self._run_test_with_price_string("1.234.567,89", 1234567.89)
        self._run_test_with_price_string("1234,567.89", 1234567.89)
        self._run_test_with_price_string("12345678.0", 12345678.0)
        self._run_test_with_price_string("12345,678.0", 12345678.0)
        self._run_test_with_price_string("12345,678.02040", 12345678.02040)

    def test_commas_only_as_thousands(self):
        """Tests prices where commas are treated as thousands separators."""
        self._run_test_with_price_string("1,000", 1000.0)
        self._run_test_with_price_string("Price is 1,234,567", 1234567.0)

        # NOTE: add logic for "," as a decimal separator if needed; default to treating "." as a decimal separator
        self._run_test_with_price_string("1,99", 199.0)  # Key test for this logic

    def test_periods_only_logic(self):
        """Tests prices with only periods."""
        self._run_test_with_price_string("1.234.567", 1234567.0)  # Multiple are thousands
        self._run_test_with_price_string("1.000", 1.0)  # Single is decimal
        self._run_test_with_price_string("Just 12.99", 12.99)  # Single is decimal
        self._run_test_with_price_string("Just 12.999987", 12.999987)  # Single is decimal

    def test_no_separators(self):
        """Tests prices with no separators."""
        self._run_test_with_price_string("$500", 500.0)
        self._run_test_with_price_string("25", 25.0)
        self._run_test_with_price_string("12345678", 12345678.0)

    def test_failure_and_edge_cases(self):
        """Tests for missing elements or non-numeric text."""
        # Test case for when the price element does not exist
        self.page.set_content("<body><p>This page has no price element.</p></body>")
        result = shopping_get_product_price(self.page)
        self.assertEqual(result, 0.0, msg="Failed on missing element")

        # Test case for an empty price string
        self._run_test_with_price_string("", 0.0)

        # Test case for text with no numbers
        self._run_test_with_price_string("On Sale!", 0.0)


# This allows the test to be run from the command line
if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
