import pytest

from npu_schedule.official import Hardware


@pytest.fixture(scope="session")
def hardware():
    return Hardware.read()
