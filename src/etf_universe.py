"""Curated diversified China ETF universe for local parquet snapshots."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EtfEntry:
    code: str
    name: str
    sector: str


# 约 75 只：宽基 + 行业 + 风格 + 主题 + 商品/债券/海外。
# 目标：本地因子 IC、截面选股（top 10%）、轮动回测有足够截面宽度。
DIVERSIFIED_ETF_UNIVERSE: tuple[EtfEntry, ...] = (
    # 宽基
    EtfEntry("510300", "沪深300ETF", "宽基"),
    EtfEntry("510500", "中证500ETF", "宽基"),
    EtfEntry("510050", "上证50ETF", "宽基"),
    EtfEntry("510180", "上证180ETF", "宽基"),
    EtfEntry("159901", "深证100ETF", "宽基"),
    EtfEntry("159915", "创业板ETF", "宽基"),
    EtfEntry("159682", "创业50ETF", "宽基"),
    EtfEntry("588000", "科创50ETF", "宽基"),
    EtfEntry("512100", "中证1000ETF", "宽基"),
    EtfEntry("563300", "中证2000ETF", "宽基"),
    # 风格 / Smart Beta
    EtfEntry("510330", "沪深300价值ETF", "风格"),
    EtfEntry("510880", "红利ETF", "风格"),
    EtfEntry("512890", "红利低波ETF", "风格"),
    EtfEntry("515100", "低波100ETF", "风格"),
    EtfEntry("159967", "创业板成长ETF", "风格"),
    # 消费
    EtfEntry("159928", "消费ETF", "消费"),
    EtfEntry("512690", "酒ETF", "消费"),
    EtfEntry("159996", "家电ETF", "消费"),
    EtfEntry("159766", "旅游ETF", "消费"),
    EtfEntry("159843", "食品饮料ETF", "消费"),
    # 医药
    EtfEntry("512010", "医药ETF", "医药"),
    EtfEntry("512170", "医疗ETF", "医药"),
    EtfEntry("159992", "创新药ETF", "医药"),
    EtfEntry("159647", "中药ETF", "医药"),
    # 金融
    EtfEntry("512880", "证券ETF", "金融"),
    EtfEntry("512000", "券商ETF", "金融"),
    EtfEntry("512800", "银行ETF", "金融"),
    EtfEntry("512070", "非银ETF", "金融"),
    EtfEntry("159841", "金融科技ETF", "金融"),
    # 科技
    EtfEntry("515000", "科技ETF", "科技"),
    EtfEntry("512760", "芯片ETF", "科技"),
    EtfEntry("159995", "芯片ETF", "科技"),
    EtfEntry("588200", "科创芯片ETF", "科技"),
    EtfEntry("512720", "计算机ETF", "科技"),
    EtfEntry("515880", "通信ETF", "科技"),
    EtfEntry("515050", "5G通信ETF", "科技"),
    EtfEntry("159819", "人工智能ETF", "科技"),
    EtfEntry("515070", "人工智能AI ETF", "科技"),
    EtfEntry("159732", "消费电子ETF", "科技"),
    EtfEntry("512330", "信息科技ETF", "科技"),
    # 新能源
    EtfEntry("516160", "新能源ETF", "新能源"),
    EtfEntry("515790", "光伏ETF", "新能源"),
    EtfEntry("515030", "新能源车ETF", "新能源"),
    EtfEntry("159790", "碳中和ETF", "新能源"),
    EtfEntry("159566", "储能ETF", "新能源"),
    # 制造 / 军工
    EtfEntry("516110", "汽车ETF", "制造"),
    EtfEntry("562500", "机器人ETF", "制造"),
    EtfEntry("159638", "高端装备ETF", "制造"),
    EtfEntry("516950", "基建ETF", "制造"),
    EtfEntry("512660", "军工ETF", "军工"),
    EtfEntry("512670", "国防ETF", "军工"),
    # 周期
    EtfEntry("512400", "有色金属ETF", "周期"),
    EtfEntry("562800", "稀有金属ETF", "周期"),
    EtfEntry("159713", "稀土ETF", "周期"),
    EtfEntry("515220", "煤炭ETF", "周期"),
    EtfEntry("515210", "钢铁ETF", "周期"),
    EtfEntry("512580", "环保ETF", "周期"),
    EtfEntry("561360", "石油ETF", "周期"),
    # 农业 / 地产 / 建材 / 公用事业
    EtfEntry("159825", "农业ETF", "农业"),
    EtfEntry("159867", "畜牧ETF", "农业"),
    EtfEntry("512200", "房地产ETF", "地产"),
    EtfEntry("159745", "建材ETF", "建材"),
    EtfEntry("561560", "电力ETF", "公用事业"),
    EtfEntry("159611", "电力ETF", "公用事业"),
    # 传媒 / 化工
    EtfEntry("512980", "传媒ETF", "传媒"),
    EtfEntry("159869", "游戏ETF", "传媒"),
    EtfEntry("516020", "化工ETF", "化工"),
    EtfEntry("159870", "化工ETF", "化工"),
    # 主题
    EtfEntry("510810", "上海国企ETF", "主题"),
    EtfEntry("512950", "央企改革ETF", "主题"),
    EtfEntry("512640", "金融地产ETF", "主题"),
    # 商品 / 债券
    EtfEntry("518880", "黄金ETF", "商品"),
    EtfEntry("159934", "黄金ETF", "商品"),
    EtfEntry("511010", "国债ETF", "债券"),
    EtfEntry("511380", "可转债ETF", "债券"),
    EtfEntry("511220", "城投债ETF", "债券"),
    # 海外
    EtfEntry("159920", "恒生ETF", "海外"),
    EtfEntry("513100", "纳指ETF", "海外"),
    EtfEntry("513500", "标普500ETF", "海外"),
    EtfEntry("513520", "日经ETF", "海外"),
    EtfEntry("513030", "德国ETF", "海外"),
    EtfEntry("513050", "中概互联ETF", "海外"),
)


def diversified_etf_symbols() -> list[str]:
    """Return diversified ETF codes in stable order."""
    return [entry.code for entry in DIVERSIFIED_ETF_UNIVERSE]


def diversified_etf_metadata() -> list[dict[str, str]]:
    """Return sector metadata for snapshot sidecars."""
    return [
        {"code": entry.code, "name": entry.name, "sector": entry.sector}
        for entry in DIVERSIFIED_ETF_UNIVERSE
    ]


def symbols_by_sector() -> dict[str, list[str]]:
    """Group diversified ETF codes by sector label."""
    grouped: dict[str, list[str]] = {}
    for entry in DIVERSIFIED_ETF_UNIVERSE:
        grouped.setdefault(entry.sector, []).append(entry.code)
    return grouped


def universe_summary() -> dict[str, int]:
    """Return counts for logging and sanity checks."""
    by_sector = symbols_by_sector()
    return {
        "symbol_count": len(DIVERSIFIED_ETF_UNIVERSE),
        "sector_count": len(by_sector),
        **{f"sector_{key}": len(value) for key, value in sorted(by_sector.items())},
    }
