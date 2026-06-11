import numpy as np
from collections import defaultdict
from statistics import mean, median


def load_triples(path):
    triples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            h, r, t = line.split("\t")
            triples.append((h, r, t))
    return triples


def compute_statistics(triples):
    entities = set()
    relations = set()

    # For |R(e)|: entity -> set of relation types
    entity_relations = defaultdict(set)

    # For relation appearance: relation -> count
    relation_counts = defaultdict(int)

    for h, r, t in triples:
        entities.add(h)
        entities.add(t)
        relations.add(r)

        entity_relations[h].add(r)
        entity_relations[t].add(r)

        relation_counts[r] += 1

    # ---- Size ----
    num_entities = len(entities)
    num_relations = len(relations)
    num_edges = len(triples)

    # ---- |R(e)| statistics ----
    r_e_values = [len(v) for v in entity_relations.values()]

    r_e_stats = {
        "Min": min(r_e_values),
        "Mean": mean(r_e_values),
        "Median": median(r_e_values),
        "Max": max(r_e_values),
    }

    # ---- Relation appearance statistics ----
    rel_app_values = list(relation_counts.values())

    rel_app_stats = {
        "Min": min(rel_app_values),
        "Mean": mean(rel_app_values),
        "Median": median(rel_app_values),
        "Max": max(rel_app_values),
    }

    return {
        "Size": {
            "#E": num_entities,
            "#R": num_relations,
            "Edges": num_edges,
        },
        "|R(e)|": r_e_stats,
        "Relation appearance": rel_app_stats,
    }


def print_table(stats):
    print("\n=== Dataset Statistics ===\n")

    print("Size")
    print(f"  #E     : {stats['Size']['#E']}")
    print(f"  #R     : {stats['Size']['#R']}")
    print(f"  Edges  : {stats['Size']['Edges']}")

    print("\n|R(e)| (number of relation types per entity)")
    for k, v in stats["|R(e)|"].items():
        print(f"  {k:<6}: {v:.2f}" if isinstance(v, float) else f"  {k:<6}: {v}")

    print("\nRelation appearance (triples per relation)")
    for k, v in stats["Relation appearance"].items():
        print(f"  {k:<6}: {v:.2f}" if isinstance(v, float) else f"  {k:<6}: {v}")


# ---------------- Example usage ----------------
if __name__ == "__main__":
    # Path to your triple file
    triple_file = "all.txt"

    triples = load_triples(triple_file)
    stats = compute_statistics(triples)
    print_table(stats)
