from ml_orchestrator.core.fitness import parse_test_results

def test_cargo_all_pass():
    out = "test result: ok. 7 passed; 0 failed; 1 ignored; 0 measured"
    assert parse_test_results(out) == {"failed": 0, "passed": 7}

def test_cargo_with_failures():
    out = "test result: FAILED. 5 passed; 2 failed; 0 ignored"
    assert parse_test_results(out) == {"failed": 2, "passed": 5}

def test_pytest_still_works():
    assert parse_test_results("== 3 failed, 10 passed in 1s ==") == {"failed": 3, "passed": 10}
