import wake_ai


def test_package_imports():
    assert hasattr(wake_ai, "AIWorkflow")


async def test_asyncio_mode_auto():
    # verifies pytest-asyncio auto mode is active (no decorator needed)
    assert True
