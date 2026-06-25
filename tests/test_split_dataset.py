from scripts.split_dataset import split_records


def test_split_records_creates_non_empty_splits():
    records = [{"id": idx} for idx in range(10)]
    train, validation = split_records(records, validation_size=0.2, seed=1)
    assert len(train) == 8
    assert len(validation) == 2
