"""Transactional disk-backed molecular records, independently keyed by method and phase."""

import json
import sqlite3
from contextlib import closing
from pathlib import Path


class MolecularRegistry:
    def __init__(self, path: Path, namespace: str, identity: dict):
        self.path = path
        self.namespace = namespace
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self.connect()) as db, db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS methods (id TEXT PRIMARY KEY, identity TEXT NOT NULL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS molecules (method TEXT, smiles TEXT, record TEXT NOT NULL, PRIMARY KEY (method,smiles))"
            )
            db.execute(
                "INSERT OR IGNORE INTO methods VALUES (?,?)",
                (namespace, json.dumps(identity, allow_nan=False)),
            )

    def connect(self):
        return sqlite3.connect(self.path, timeout=30)

    def get(self, smiles):
        with closing(self.connect()) as db:
            row = db.execute(
                "SELECT record FROM molecules WHERE method=? AND smiles=?",
                (self.namespace, smiles),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def __setitem__(self, smiles, record):
        with closing(self.connect()) as db, db:
            db.execute(
                "INSERT OR REPLACE INTO molecules VALUES (?,?,?)",
                (self.namespace, smiles, json.dumps(record, allow_nan=False)),
            )

    def __delitem__(self, smiles):
        with closing(self.connect()) as db, db:
            db.execute(
                "DELETE FROM molecules WHERE method=? AND smiles=?",
                (self.namespace, smiles),
            )
