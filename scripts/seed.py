from app.core.config import get_settings
from app.db.seed import seed_all
from app.services.container import ApplicationContainer


def main() -> None:
    container = ApplicationContainer(get_settings())
    try:
        container.database.create_schema()
        print(seed_all(container.database, container.rag))
    finally:
        container.close()


if __name__ == "__main__":
    main()

