__test__ = False


def main():
    import ScraperFC

    fbref = ScraperFC.FBref()
    try:
        print("Scraping UEFA Champions League 2023...")
        df = fbref.scrape_matches(year=2023, league="UEFA Champions League")
        print("Available columns:", list(df.columns))
        # Print a sample row
        print("Sample row:")
        print(df.head(1).T)
        
        # Save sample to check
        df.head(10).to_csv('cl_sample.csv', index=False)
        print("Saved sample to cl_sample.csv")
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
