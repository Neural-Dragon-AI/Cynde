from typing import List, Union
import polars as pl

def prompt(df: pl.DataFrame, fstring: str, expressions: List[Union[pl.Expr, str]], prompt_name: str, context: str = 'with_columns') -> pl.DataFrame:
    """
    Dynamically generates prompts based on the given format string and expressions, 
    and either adds them as a new column to the DataFrame or selects them based on the specified context.

    Parameters:
    - df: The Polars DataFrame to which the prompts will be added or from which data will be selected.
    - fstring: A format string with placeholders for the expressions. If a plain string value is to be included, 
               it will be converted to a Polars expression.
    - expressions: A list of Polars expressions or string literals. Each expression must result in either a scalar value 
                   or a list of values all having the same length. When using 'with_columns' context, the expressions 
                   must return lists of the same length as the full DataFrame.
    - prompt_name: The name of the new column that will contain the generated prompts.
    - context: A string indicating the operation context. Valid values are 'with_columns' and 'select'.
               'with_columns' appends the generated prompts as a new column, requiring list results to match
               the DataFrame length. 'select' creates a new DataFrame from the generated prompts, potentially
               alongside other specified columns.

    Returns:
    - A DataFrame with the added prompts column if 'with_columns' is used, or a new DataFrame with selected columns
      if 'select' is used. The result of each expression used in the formatting must result in either a scalar 
      value or a list of values all having the same length, especially for 'with_columns' context.
    """
    # Convert string values in expressions to Polars expressions
    expressions = [pl.lit(expr) if isinstance(expr, str) else expr for expr in expressions]

    # Validate inputs
    if not isinstance(df, pl.DataFrame):
        raise ValueError("df must be a Polars DataFrame.")
    if not isinstance(fstring, str):
        raise ValueError("fstring must be a string.")
    if not all(isinstance(expr, pl.Expr) for expr in expressions):
        raise ValueError("All items in expressions must be Polars Expr or string literals converted to Polars expressions.")
    if not isinstance(prompt_name, str):
        raise ValueError("prompt_name must be a string.")
    if context not in ['with_columns', 'select']:
        raise ValueError("context must be either 'with_columns' or 'select'.")

    # Validate the number of placeholders matches the number of expressions
    placeholders_count = fstring.count("{}")
    if placeholders_count != len(expressions):
        raise ValueError(f"The number of placeholders in fstring ({placeholders_count}) does not match the number of expressions ({len(expressions)}).")

    # Use pl.format to generate the formatted expressions
    formatted_expr = pl.format(fstring, *expressions).alias(prompt_name)
    
    # Apply the context-specific operation
    if context == 'with_columns':
        # Append the generated prompt as a new column
        return df.with_columns(formatted_expr)
    else:  # context == 'select'
        # Create a new DataFrame with only the generated prompts or alongside other specified columns
        return df.select(formatted_expr)
