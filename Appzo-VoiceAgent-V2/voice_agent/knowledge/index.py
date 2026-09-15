from dataclasses import dataclass
import re


@dataclass(frozen=True)
class KnowledgeRecord:
    tenant_id: str; agent_id: str; knowledge_version: str; document_id: str; text: str; risk_class: str = "LOW_PUBLIC"


class TenantKnowledgeIndex:
    """In-memory, tenant-keyed retrieval. Replace the scorer, not this boundary."""
    def __init__(self) -> None: self._records: dict[tuple[str, str, str], list[KnowledgeRecord]] = {}

    def add(self, record: KnowledgeRecord) -> None:
        self._records.setdefault((record.tenant_id, record.agent_id, record.knowledge_version), []).append(record)

    def search(self, *, tenant_id: str, agent_id: str, knowledge_version: str, query: str, top_k: int = 3) -> list[KnowledgeRecord]:
        words = set(re.findall(r"\w+", query.lower()))
        records = self._records.get((tenant_id, agent_id, knowledge_version), [])
        ranked = [
            (len(words & set(re.findall(r"\w+", record.text.lower()))), record)
            for record in records
        ]
        return [record for score, record in sorted(ranked, key=lambda item: item[0], reverse=True)[:top_k] if score]
