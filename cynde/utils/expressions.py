import polars as pl

def list_struct_to_string(col_name: str, separator: str = " ") -> pl.Expr:
    return pl.col(col_name).list.eval(pl.element().struct.json_encode()).list.join(separator=separator).alias(f"str_{col_name}")
