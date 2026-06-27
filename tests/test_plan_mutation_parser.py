from src.plan_mutation import parse_plan_mutation
from src.research_plan import MutationType


def _context(selected_factor_names=None, selected_factors=None, target=None):
    return {
        "selected_factor_names": selected_factor_names or [],
        "selected_factors": selected_factors or [],
        "target": target or {"metric": "future_return", "horizon": 5},
        "committed_plan": {
            "selected_factor_names": selected_factor_names or [],
            "target": target or {"metric": "future_return", "horizon": 5},
            "selection_source": "explicit",
            "selection_status": "ready",
        },
    }


def test_parse_add_factor():
    mutation = parse_plan_mutation("再加上低波动", _context(selected_factor_names=["momentum_20d"]))
    assert mutation.mutation_type == MutationType.ADD_FACTORS
    assert "volatility_20d" in mutation.add_factor_names


def test_parse_remove_factor():
    mutation = parse_plan_mutation(
        "去掉动量",
        _context(
            selected_factor_names=["momentum_20d", "volatility_20d"],
            selected_factors=[
                {"name": "momentum_20d", "category": "momentum"},
                {"name": "volatility_20d", "category": "risk"},
            ],
        ),
    )
    assert mutation.mutation_type == MutationType.REMOVE_FACTORS
    assert mutation.remove_factor_names == ["momentum_20d"]


def test_parse_replace_factor():
    mutation = parse_plan_mutation("不要20日动量，改成60日动量", _context(selected_factor_names=["momentum_20d"]))
    assert mutation.mutation_type == MutationType.REPLACE_FACTORS
    assert mutation.remove_factor_names == ["momentum_20d"]
    assert mutation.add_factor_names == ["momentum_60d"]


def test_parse_target_update():
    mutation = parse_plan_mutation("那未来10日呢？", _context(selected_factor_names=["momentum_20d"]))
    assert mutation.mutation_type == MutationType.UPDATE_TARGET
    assert mutation.target_horizon == 10
    assert mutation.target_metric == "future_return"


def test_parse_reset_plan():
    mutation = parse_plan_mutation("重新研究成交额放大", _context(selected_factor_names=["momentum_20d"]))
    assert mutation.mutation_type == MutationType.RESET_PLAN


def test_parse_no_op():
    mutation = parse_plan_mutation("你好", _context())
    assert mutation.mutation_type == MutationType.NO_OP
from src.plan_mutation import parse_plan_mutation
from src.research_plan import MutationType


def _context(selected_factor_names=None, selected_factors=None, target=None):
    return {
        "selected_factor_names": selected_factor_names or [],
        "selected_factors": selected_factors or [],
        "target": target or {"metric": "future_return", "horizon": 5},
        "committed_plan": {
            "selected_factor_names": selected_factor_names or [],
            "target": target or {"metric": "future_return", "horizon": 5},
            "selection_source": "explicit",
            "selection_status": "ready",
        },
    }


def test_parse_add_factor():
    mutation = parse_plan_mutation("再加上低波动", _context(selected_factor_names=["momentum_20d"]))
    assert mutation.mutation_type == MutationType.ADD_FACTORS
    assert "volatility_20d" in mutation.add_factor_names


def test_parse_remove_factor():
    mutation = parse_plan_mutation(
        "去掉动量",
        _context(
            selected_factor_names=["momentum_20d", "volatility_20d"],
            selected_factors=[
                {"name": "momentum_20d", "category": "momentum"},
                {"name": "volatility_20d", "category": "risk"},
            ],
        ),
    )
    assert mutation.mutation_type == MutationType.REMOVE_FACTORS
    assert mutation.remove_factor_names == ["momentum_20d"]


def test_parse_replace_factor():
    mutation = parse_plan_mutation("不要20日动量，改成60日动量", _context(selected_factor_names=["momentum_20d"]))
    assert mutation.mutation_type == MutationType.REPLACE_FACTORS
    assert mutation.remove_factor_names == ["momentum_20d"]
    assert mutation.add_factor_names == ["momentum_60d"]


def test_parse_target_update():
    mutation = parse_plan_mutation("那未来10日呢？", _context(selected_factor_names=["momentum_20d"]))
    assert mutation.mutation_type == MutationType.UPDATE_TARGET
    assert mutation.target_horizon == 10
    assert mutation.target_metric == "future_return"


def test_parse_reset_plan():
    mutation = parse_plan_mutation("重新研究成交额放大", _context(selected_factor_names=["momentum_20d"]))
    assert mutation.mutation_type == MutationType.RESET_PLAN


def test_parse_no_op():
    mutation = parse_plan_mutation("你好", _context())
    assert mutation.mutation_type == MutationType.NO_OP
