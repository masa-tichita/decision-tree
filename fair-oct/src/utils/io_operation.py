import polars as pl


def read_csv(file_path: str) -> pl.LazyFrame:
    """Read a CSV file and return a lazy frame
    Args:
        file_path (str): Path to the CSV file
    Returns:
        pl.LazyFrame: A lazy frame
    """
    return pl.read_csv(file_path).lazy()
