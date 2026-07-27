from ml_orchestrator.core.fitness import parse_test_results

def test_mocha_all_pass():
    assert parse_test_results("  12 passing (340ms)") == {"failed": 0, "passed": 12}

def test_mocha_with_failures():
    out = "  10 passing (2s)\n  3 failing"
    assert parse_test_results(out) == {"failed": 3, "passed": 10}

def test_pytest_unbroken():
    assert parse_test_results("== 3 failed, 10 passed in 1s ==") == {"failed": 3, "passed": 10}
