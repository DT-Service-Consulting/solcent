import pytest


@pytest.mark.skip(reason="network test; run manually for the data spike")
def test_elia_fetch_smoke():
    from datetime import datetime
    from ingest import elia
    df = elia.fetch_ods032(datetime(2025, 4, 19), datetime(2025, 4, 19))
    assert not df.empty
    assert "measured" in df.columns
