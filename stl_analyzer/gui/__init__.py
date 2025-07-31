"""
STL Analyzer GUI Package
"""

def run_app():
    """Run the STL Analyzer GUI application using modular structure"""
    from .main_window import run_app as main_run_app
    return main_run_app()

__all__ = ['run_app'] 