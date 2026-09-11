from churn.synthesize import FEATURES, generate


def test_reproducible_from_seed():
    a = generate(500, seed=42)
    b = generate(500, seed=42)
    assert a.equals(b)


def test_shape_and_columns():
    df = generate(1000, seed=1)
    assert len(df) == 1000
    for col in FEATURES + ["churned"]:
        assert col in df.columns
    assert not df[FEATURES].isna().any().any()


def test_label_is_not_trivial():
    df = generate(2000, seed=3)
    rate = df["churned"].mean()
    assert 0.15 < rate < 0.6   # realistic churn band, not degenerate
