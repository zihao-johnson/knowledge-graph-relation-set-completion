from collections import Counter, defaultdict
import numpy as np
from copy import deepcopy
import os

def parse_relation_set_lines(
    lines,
    sep=",",
    dedup_within_line=True,
):
    """
    lines: list[str]
    each line: entity,r1,r2,...
    returns: list of (entity, set(relations))
    """
    data = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(sep) if p.strip()]
        if len(parts) < 1:
            continue
        entity = parts[0]
        rels = parts[1:]
        if dedup_within_line:
            rels = set(rels)
        data.append((entity, rels))
    return data

def compute_dataset_statistics(data):
    """
    data: list of (entity, set(relations))
    returns: stats dict + artifacts
    """
    ent2rels = defaultdict(set)
    for e, rs in data:
        ent2rels[e].update(rs)

    entities = list(ent2rels.keys())
    num_entities = len(entities)

    # per-entity relation count
    degrees = np.array([len(ent2rels[e]) for e in entities])

    # relation appearance (how many entities contain it)
    rel_freq = Counter()
    for rs in ent2rels.values():
        rel_freq.update(rs)

    rel_counts = np.array(list(rel_freq.values()))
    num_relations = len(rel_freq)

    stats = {
        "num_entities": num_entities,
        "num_relations": num_relations,
        "total_entity_relation_edges": int(degrees.sum()),

        "entity_degree_min": int(degrees.min()),
        "entity_degree_mean": float(degrees.mean()),
        "entity_degree_median": float(np.median(degrees)),
        "entity_degree_p95": float(np.percentile(degrees, 95)),
        "entity_degree_max": int(degrees.max()),

        "relation_freq_min": int(rel_counts.min()),
        "relation_freq_mean": float(rel_counts.mean()),
        "relation_freq_median": float(np.median(rel_counts)),
        "relation_freq_p95": float(np.percentile(rel_counts, 95)),
        "relation_freq_max": int(rel_counts.max()),

        "relation_singletons": int((rel_counts == 1).sum()),
        "relation_leq_5": int((rel_counts <= 5).sum()),
    }

    artifacts = {
        "ent2rels": ent2rels,
        "degrees": degrees,
        "relation_freq": rel_freq,
    }

    return stats, artifacts

def write_stats_to_txt(stats, rel_freq, out_path, topk_rel=30):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("=== Dataset Statistics (Relation-Set Format) ===\n\n")

        f.write(f"#Entities: {stats['num_entities']}\n")
        f.write(f"#Relations: {stats['num_relations']}\n")
        f.write(f"#Entity–Relation edges: {stats['total_entity_relation_edges']}\n\n")

        f.write("Entity degree |R(e)|:\n")
        f.write(
            f"  min={stats['entity_degree_min']}, "
            f"mean={stats['entity_degree_mean']:.2f}, "
            f"median={stats['entity_degree_median']:.2f}, "
            f"p95={stats['entity_degree_p95']:.2f}, "
            f"max={stats['entity_degree_max']}\n\n"
        )

        f.write("Relation appearance (#entities per relation):\n")
        f.write(
            f"  min={stats['relation_freq_min']}, "
            f"mean={stats['relation_freq_mean']:.2f}, "
            f"median={stats['relation_freq_median']:.2f}, "
            f"p95={stats['relation_freq_p95']:.2f}, "
            f"max={stats['relation_freq_max']}\n"
        )
        f.write(f"  #singleton relations: {stats['relation_singletons']}\n")
        f.write(f"  #relations ≤5: {stats['relation_leq_5']}\n\n")

        f.write(f"Top-{topk_rel} relations:\n")
        for r, c in rel_freq.most_common(topk_rel):
            f.write(f"{r}\t{c}\n")


def plot_entity_degree_hist(degrees, path):
    import matplotlib.pyplot as plt
    plt.figure()
    bins = np.arange(0, degrees.max() + 2)
    plt.hist(degrees, bins=bins)
    plt.xlabel("|R(e)|")
    plt.ylabel("#Entities")
    plt.title("Entity relation-set size distribution")
    plt.tight_layout()
    plt.savefig(os.path.join("statis","entity_"+path))


def plot_top_er_heatmap(ent2rels, rel_freq, path, top_entities=30, top_relations=30):
    import matplotlib.pyplot as plt

    # top relations
    rels = [r for r, _ in rel_freq.most_common(top_relations)]
    rid = {r: i for i, r in enumerate(rels)}

    # top entities by degree
    ent_sorted = sorted(ent2rels.items(), key=lambda x: len(x[1]), reverse=True)
    ents = [e for e, _ in ent_sorted[:top_entities]]

    M = np.zeros((len(ents), len(rels)))
    for i, e in enumerate(ents):
        for r in ent2rels[e]:
            if r in rid:
                M[i, rid[r]] = 1

    plt.figure(figsize=(8, 6))
    plt.imshow(M, aspect="auto")
    plt.colorbar(label="Incidence")
    plt.xlabel("Relations")
    plt.ylabel("Entities")
    plt.title("Top entity–relation incidence heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join("statis","heatmap_"+path))


def plot_relation_zipf(rel_freq, path, max_points=2000):
    import matplotlib.pyplot as plt
    freqs = np.array(sorted(rel_freq.values(), reverse=True))[:max_points]
    plt.figure()
    plt.plot(np.arange(1, len(freqs) + 1), freqs)
    plt.xlabel("Relation rank")
    plt.ylabel("Frequency (#entities)")
    plt.title("Relation frequency (Zipf-like)")
    plt.tight_layout()
    plt.savefig(os.path.join("statis","relations_"+path))

from collections import Counter
from typing import Dict, Union, Iterable, Optional, Tuple
import numpy as np

from collections import Counter
from typing import Dict, Union, Optional, Tuple
import numpy as np

def plot_relation_freq_three_splits(
    train_freq: Union[Counter, Dict[str, int]],
    valid_freq: Union[Counter, Dict[str, int]],
    test_freq:  Union[Counter, Dict[str, int]],
    *,
    top_n: Optional[int] = 200,
    sort_by: str = "train",          # "train" or "total"
    log_y: bool = True,
    normalize: bool = False,
    title: str = "",
    xlabel: str = "Relation type (ranked)",
    ylabel: str = "Frequency",
    legend_loc: str = "best",
    figsize: Tuple[float, float] = (7.5, 4.2),
    linewidth: float = 1.0,
    line_alpha: float = 0.95,
    fill_alphas: Tuple[float, float, float] = (0.18, 0.12, 0.10),  # Train/Valid/Test
    save_path: Optional[str] = None,  # e.g., "rel_freq.pdf"
    dpi: int = 300,
) -> None:
    """
    - Missing relations in a split are automatically assigned frequency 0.
    - Draws 3 lines + filled areas under each curve (different transparency).
    """
    import matplotlib.pyplot as plt

    tr = Counter(train_freq)
    va = Counter(valid_freq)
    te = Counter(test_freq)

    vocab = set(tr) | set(va) | set(te)

    if sort_by == "train":
        order = sorted(vocab, key=lambda r: tr.get(r, 0), reverse=True)
    elif sort_by == "total":
        order = sorted(vocab, key=lambda r: (tr.get(r, 0) + va.get(r, 0) + te.get(r, 0)), reverse=True)
    else:
        raise ValueError("sort_by must be 'train' or 'total'")

    if top_n is not None:
        order = order[:top_n]

    # Missing => 0 (via .get)
    y_tr = np.array([tr.get(r, 0) for r in order], dtype=np.float64)
    y_va = np.array([va.get(r, 0) for r in order], dtype=np.float64)
    y_te = np.array([te.get(r, 0) for r in order], dtype=np.float64)

    if normalize:
        y_tr = y_tr / max(1.0, y_tr.sum())
        y_va = y_va / max(1.0, y_va.sum())
        y_te = y_te / max(1.0, y_te.sum())
        ylabel_plot = "Proportion"
    else:
        ylabel_plot = ylabel

    x = np.arange(1, len(order) + 1)

    plt.figure(figsize=figsize, dpi=dpi)
    plt.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.5)

    # Plot lines first to get their default colors
    line_tr, = plt.plot(x, y_tr, label="Train", linewidth=linewidth, alpha=line_alpha)
    line_va, = plt.plot(x, y_va, label="Valid", linewidth=linewidth, alpha=line_alpha)
    line_te, = plt.plot(x, y_te, label="Test",  linewidth=linewidth, alpha=line_alpha)

    # Fill under curves using the same colors as lines
    plt.fill_between(x, y_tr, 0.0, alpha=fill_alphas[0], color=line_tr.get_color())
    plt.fill_between(x, y_va, 0.0, alpha=fill_alphas[1], color=line_va.get_color())
    plt.fill_between(x, y_te, 0.0, alpha=fill_alphas[2], color=line_te.get_color())

    if log_y:
        # log scale can't show y=0; clamp zeros to a tiny epsilon for plotting only
        eps = 1e-12
        plt.yscale("log")
        plt.ylabel("",fontsize=20)
        # IMPORTANT: fill_between already drew; log scale affects rendering.
        # If you have many zeros and want nicer visuals, set log_y=False or normalize=True.
    else:
        plt.ylabel("",fontsize=20)

    plt.xlabel(xlabel, fontsize=20)
    plt.title(title)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(loc=legend_loc, frameon=True, fontsize=20)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    

if __name__ == "__main__":
    
    paths = ["train.txt", "valid.txt", "test.txt", "rest.txt"]
    fre = []
    data_all = []
    for path in paths:

        print("For ", path)
        with open(path,"r") as file:
            data = file.readlines()
        print(data[0])

        print("num lines:", len(data))
        data = parse_relation_set_lines(data, sep="	")
        print("parsed rows:", len(data))
        print("first 3 rows:", data[0])
        print("avg rels per row:", sum(len(rs) for _, rs in data) / max(1, len(data)))
        data_all.append(data)

        stats, artifacts = compute_dataset_statistics(data)

        write_stats_to_txt(
            stats,
            artifacts["relation_freq"],
            out_path=os.path.join("statis",path.split(",")[0]+"_dataset_stats.txt")
        )
        path = path.split(",")[0]+".png"
        plot_entity_degree_hist(artifacts["degrees"],path)
        plot_relation_zipf(artifacts["relation_freq"],path=path)
        plot_top_er_heatmap(
            artifacts["ent2rels"],
            artifacts["relation_freq"],
            path=path
        )
        fre.append(artifacts["relation_freq"])
    # train_rel_freq = Counter(...)  # relation -> count
    # valid_rel_freq = Counter(...)
    # test_rel_freq  = Counter(...)

    plot_relation_freq_three_splits(
        fre[0], fre[1], fre[2],
        top_n=300,          # show top 300 relations
        sort_by="train",
        log_y=True,
        normalize=False,
        save_path="relation_freq_splits.pdf",  # high-quality for ICML
    )



    # merge datasets:
    all_data = deepcopy(data_all[0])
    print(len(all_data))
    print(type(all_data[0][1]))
    all_data = {e[0]:list(e[1:]) for e in data_all[0]}

    for i in [1,2]:
        for e in data_all[i]:
            all_data[e[0]] += list(e[1:])
    for k,v in all_data.items():
        print(k, v)
        break
    all_data = [[key, value] for key, value in all_data.items()]
    
    print(len(all_data))
    print(type(all_data[0][1]))
    stats, artifacts = compute_dataset_statistics(data)

    write_stats_to_txt(
        stats,
        artifacts["relation_freq"],
        out_path=os.path.join("statis_all_dataset_stats.txt")
    )
