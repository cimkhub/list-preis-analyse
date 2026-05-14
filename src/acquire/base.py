import logging
from abc import ABC, abstractmethod

from src.models import AcquiredDocument, RawProduct

logger = logging.getLogger("birkenhof.acquire")


class BaseScraper(ABC):
    def __init__(self, config: dict, storage_base: str = "data"):
        self.config = config
        self.storage_base = storage_base
        self.logger = logging.getLogger(f"birkenhof.acquire.{self.supplier_name}")

    @property
    @abstractmethod
    def supplier_name(self) -> str:
        pass

    @abstractmethod
    def get_current_offers(self, week: int, year: int, force: bool = False) -> list[AcquiredDocument]:
        pass

    @abstractmethod
    def extract_products(self, document: AcquiredDocument) -> list[RawProduct]:
        pass
