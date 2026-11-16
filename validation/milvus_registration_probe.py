"""Observe repeated insert/upsert writes against native Milvus Lite."""
import json
import tempfile
from pathlib import Path
from pymilvus import MilvusClient

with tempfile.TemporaryDirectory() as directory:
    client = MilvusClient(uri=str(Path(directory) / "probe.db"))
    try:
        for method in ("insert", "upsert"):
            collection = f"registration_{method}"
            client.create_collection(
                collection_name=collection,
                dimension=2,
                auto_id=False,
                consistency_level="Strong",
            )
            write = getattr(client, method)
            for label in ("first.jpg", "latest.jpg"):
                result = write(
                    collection_name=collection,
                    data=[{"id": 7, "vector": [1.0, 0.0], "img_name": label}],
                )
                print(method, label, result, flush=True)
            client.flush(collection_name=collection)
            print(method, "count", client.query(
                collection_name=collection, filter="", output_fields=["count(*)"]
            ), flush=True)
            print(method, "query", client.query(
                collection_name=collection, filter="id >= 0", output_fields=["id", "img_name"]
            ), flush=True)
            print(method, "integer id", client.get(collection_name=collection, ids=[7]), flush=True)
            try:
                print(method, "string id", client.get(
                    collection_name=collection, ids=["7"]
                ), flush=True)
            except Exception as error:
                print(method, "string id failed", type(error).__name__, str(error), flush=True)
            print(method, "search", client.search(
                collection_name=collection, data=[[1.0, 0.0]], limit=10,
                output_fields=["img_name"]
            ), flush=True)
        collection = "within_batch"
        client.create_collection(collection_name=collection, dimension=2, auto_id=False)
        data = [{"id": 7, "vector": [1.0, 0.0], "img_name": name}
                for name in ("first.jpg", "latest.jpg")]
        print("within_batch", client.upsert(collection, data), flush=True)
        client.flush(collection)
        print("within_batch count", client.query(
            collection, filter="", output_fields=["count(*)"]
        ), flush=True)
    finally:
        client.close()
