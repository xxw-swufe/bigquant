from src.factor_generator import generate_factor_candidates
from src.etf_factor_library import get_factor_library


def test_generated_factors_do_not_use_future_data():
    factors = generate_factor_candidates({})
    assert factors
    assert all(not factor["uses_future_data"] for factor in factors)
    assert all("m_lead" not in factor["formula"] for factor in factors)


def test_reference_library_size_is_mvp_friendly():
    factors = get_factor_library()
    assert 10 <= len(factors) <= 40
    assert all(not factor["uses_future_data"] for factor in factors)
