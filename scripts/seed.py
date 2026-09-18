"""手动初始化数据库的命令行入口。

数据流：``python scripts/seed.py`` → Settings → Container → 建表 → seed_all → 统计输出。
"""

from app.core.config import get_settings
from app.db.seed import seed_all
from app.services.container import ApplicationContainer


def main() -> None:
    """建立完整依赖后执行一次幂等播种，并确保连接最终关闭。"""

    container = ApplicationContainer(get_settings())
    try:
        container.database.create_schema()
        print(seed_all(container.database, container.rag))
    finally:
        container.close()


if __name__ == "__main__":
    main()
