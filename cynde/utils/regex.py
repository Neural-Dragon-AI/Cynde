import re

def format_string_to_regex(fstring, var_names):
    # Escape regex special characters in the static parts of the format string
    escaped_string = re.escape(fstring)
    
    # Prepare a regex pattern for named capturing groups
    for var_name in var_names:
        # Replace the first occurrence of escaped "{}" with a named capturing group
        escaped_string = re.sub(r'\\\{\\\}', f'(?P<{var_name}>[^_]+)', escaped_string, count=1)
    
    return escaped_string

def parse_format_string(input_string, fstring, var_names):
    # Generate the regex pattern from the format string and variable names
    regex_pattern = format_string_to_regex(fstring, var_names)
    
    # Attempt to match the regex pattern against the input string
    match = re.search(regex_pattern, input_string)
    
    if match:
        # Extract matched values into a dictionary using the variable names
        result = {var_name: match.group(var_name) for var_name in var_names}
        return result
    else:
        # Return None or an empty dictionary if no match is found
        return None