def print_colored(text, style="info"):
    """
    Print colored text to the console for better readability.

    Parameters:
    - text (str): The message to print.
    - style (str): One of 'info', 'warn', or 'error'.
    """
    colors = {
        "info": "\033[94m",    # Blue
        "warn": "\033[93m",    # Yellow
        "error": "\033[91m",   # Red
        "gray": "\033[90m",    # Gray
        "end": "\033[0m",      # Reset
    }

    color = colors.get(style, colors["info"])
    print(f"{color}{text}{colors['end']}")
