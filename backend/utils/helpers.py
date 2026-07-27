"""
Utility helper functions module.

Provides shared operations such as JSON manipulation, format conversions,
and mathematical models calculations.
"""

from typing import Any, Dict


def format_currency(amount: float) -> str:
    """
    Format a float value into USD currency string format.

    Args:
        amount: Numerical cash amount.

    Returns:
        str: Formatted currency representation (e.g. '$1,250.50').
    """
    return f"${amount:,.2f}"


def calculate_margin(revenue: float, cost: float) -> float:
    """
    Calculate margins of profitability.

    Args:
        revenue: Total revenue earned.
        cost: Costs incurred.

    Returns:
        float: Margin percentage representation (0.0 to 1.0).
    """
    if revenue <= 0.0:
        return 0.0
    return (revenue - cost) / revenue


def clean_dict_keys(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format keys of a dictionary to snake_case format.

    Args:
        data: Input dictionary.

    Returns:
        Dict[str, Any]: Cleaned dictionary.
    """
    return {
        key.strip().lower().replace(" ", "_").replace("-", "_"): val
        for key, val in data.items()
    }
