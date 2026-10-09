"""Names and settings shared by the pipeline, notebooks and tests."""

from __future__ import annotations

from dataclasses import dataclass

# Raw file stem -> bronze/silver table name.
RAW_FILES: dict[str, str] = {
    "transaction_data": "transactions",
    "causal_data": "causal",
    "product": "products",
    "hh_demographic": "demographics",
    "campaign_desc": "campaign_desc",
    "campaign_table": "campaign_table",
    "coupon": "coupons",
    "coupon_redempt": "coupon_redemptions",
}

LAYERS = ("bronze", "silver", "gold", "ops")

# Fuel and other non-merchandise lines. Quantity on these is in non-unit measures
# (e.g. 89,638 at $0.0023 each), so they are excluded from price and unit analysis.
NON_MERCH_DEPARTMENTS = frozenset({"KIOSK-GAS", "MISC SALES TRAN", "COUP/STR & MFG"})
NON_MERCH_COMMODITIES = frozenset({"COUPON/MISC ITEMS"})

# Documented promotion codes (user guide, causal_data).
DISPLAY_CODES = frozenset({"0", "1", "2", "3", "4", "5", "6", "7", "9", "A"})
MAILER_CODES = frozenset({"0", "A", "C", "D", "F", "H", "J", "L", "P", "X", "Z"})

LAST_DAY = 711  # last transaction day in the data


@dataclass(frozen=True)
class Settings:
    catalog: str = "retail"
    panel_first_week: int = 9  # causal_data covers weeks 9-101
    panel_last_week: int = 101
    panel_min_weeks: int = 26  # product must be bought in at least this many weeks
    raw_volume_dir: str | None = None  # defaults to /Volumes/<catalog>/raw/landing

    @property
    def landing_dir(self) -> str:
        return self.raw_volume_dir or f"/Volumes/{self.catalog}/raw/landing"
