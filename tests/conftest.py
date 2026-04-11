def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "unity: marks tests that require a running Unity simulator (deselect with -m 'not unity')",
    )
