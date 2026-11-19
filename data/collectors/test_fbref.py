"""Standalone FBref Champions League smoke scraper utility."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

__test__ = False


def scrape_cl_sample(
    year: int = 2023,
    league: str = "UEFA Champions League",
    output_csv: str | Path = "cl_sample.csv",
    fbref_client: Optional[Any] = None,
):
    """Fetch sample matches from FBref (or an injected mock client) and persist top 10 rows."""
    if fbref_client is None:
        import ScraperFC  # pylint: disable=import-outside-toplevel

        fbref_client = ScraperFC.FBref()

    print(f"Scraping {league} {year}...")
    df = fbref_client.scrape_matches(year=year, league=league)
    print("Available columns:", list(df.columns))
    print("Sample row:")
    print(df.head(1).T)

    output_path = Path(output_csv)
    df.head(10).to_csv(output_path, index=False)
    print(f"Saved sample to {output_path}")
    return df


def main() -> None:
    """CLI entrypoint for quick manual FBref connectivity verification."""
    try:
        scrape_cl_sample()
    except Exception as exc:
        print("Error:", exc)


if __name__ == "__main__":
    main()
