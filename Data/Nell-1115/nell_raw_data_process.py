import pandas as pd 
import sys
import numpy as np
import argparse
import tqdm
import networkx as nx
from datetime import datetime
import random

def load_data(path):
    df = pd.read_csv(path, encoding = "utf-8", sep = "\t", on_bad_lines="skip")
    print(f"Original data contains {len(df)} lines")
    df = df.sort_values(by="Probability", ascending = False)
    df = df[df["Relation"] != "generalizations"]
    print(f"Data without generalizations entries contains {len(df)} lines")
    df = df.filter(items = ["Entity", "Relation", "Value"])
    return df.values.tolist()

def save_processed_data(path, data):
    with open(path, "w", encoding="utf-8") as file:
        file.writelines("\n".join(["\t".join(e) for e in data]))

def find_connected_subgraphs(data):
    G = nx.MultiDiGraph()
    G.add_edges_from([[e[0], e[2], e[1]] for e in data])   # data = list of (head, tail, relation)
    components = list(nx.weakly_connected_components(G))
    sub_triplets = [[(h, r, t) for h, t, r in G.subgraph(component).edges(keys=True)] for component in components]
    comp_stats = []
    original_subgraphs_No = sum([len(sub_triplet) for sub_triplet in sub_triplets])

    for i, nodes in enumerate(components):
        subG = G.subgraph(nodes)
        avg_deg = sum(dict(subG.degree()).values()) / len(subG) 
        edge_count = subG.number_of_edges()
        node_count = len(subG)
        comp_stats.append({
            "component_id": i,
            "nodes": node_count,
            "edges": edge_count,
            "avg_degree": avg_deg,
            "relations_No" : len(set([e[1] for e in sub_triplets[i]])),
            "entity_No" : len(set([e[0] for e in sub_triplets[i]] + [e[2] for e in sub_triplets[i]])),
            "triplets_No" : len(sub_triplets[i])

        })
    print(f"No of subgraphs {len(comp_stats)}")
    
    while True:
        ans = input("Check subgraphs with average more than: ")
        selected = [e for e in comp_stats if e["avg_degree"] > float(ans)]
        ans = input(f"No of selected subgraphs {len(selected)}, continue print them (y or n)?: ")
        if ans.lower() == "y":
            for e in selected:
                print(e)
        else:
            pass
        ans = input("check subgraphs? (Y or N)")
        if ans.lower() == "y":
            continue
        else:
            break

    ans = input(f"Which subgraph should be selected? enter from {0} to {len(sub_triplets)}, seperated by ,: ")
    index = [int(e) for e in ans.split(",")]
    return [components[e] for e in index], [sub_triplets[e] for e in index], [comp_stats[e] for e in index], original_subgraphs_No

def pure_graphs(data, k = 5, sample_No = 30000):
    G = nx.MultiDiGraph()
    G.add_edges_from([[e[0], e[2], e[1]] for e in data])
    G_filtered = G.copy()
    while True:
        def in_edge(n): 
            return [r for _,_,r in G_filtered.in_edges(n, keys=True)]
        def out_edge(n): 
            return [r for _,_,r in G_filtered.out_edges(n, keys=True)]
        remains_node = [n for n in G_filtered.nodes() if len(set(in_edge(n) + out_edge(n))) > k]
        triplets = []
        for n in remains_node:
            triplets.extend([(u, r, v) for u, v, r in G_filtered.edges(n, keys=True)])
            if len(triplets) >= sample_No:
                break
        # Print summary
        print(f"Remained triplets: {len(triplets)}, remained entities: {len(set([e[0] for e in triplets] + [e[2] for e in triplets]))}, remained relations: {len(set(e[1] for e in triplets))}")
        argin = input(f"Define you k (y/n)?: ")
        if argin.lower() == "y":
            while True:
                argin=input(f"Define you k number: ")
                try:
                    k = int(argin)
                    break
                except:
                    print("Invalid k value, enter again...")
                    continue
            continue
        else:
            graphs, sub_triplets, statistics, original_subgraphs_No = find_connected_subgraphs(data = triplets)
            sub_triplets = [triplet for graph in sub_triplets for triplet in graph]
            return sub_triplets, statistics
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="My script with defaults")

    parser.add_argument("--path", default="NELL.08m.1115.esv.csv", help="Input file name")
    parser.add_argument("--save_path", default="all.txt", help="Clearned dataset")
    parser.add_argument("--lines_No", default=50000, help="No of saved data entries")

    args = parser.parse_args()

    original_triplets = load_data(args.path)
    No_relations = len(list(set([e[1] for e in original_triplets])))
    No_entity = len(list(set([e[0] for e in original_triplets] + [e[2] for e in original_triplets])))
    print(f"ALL Numnber of entity: {No_entity}, ALL number of relations: {No_relations}, ALL number of triplets: {len(original_triplets)}\n")
    
    sub_triplets,statistics = pure_graphs(original_triplets,sample_No=args.lines_No)
    
    save_processed_data(args.save_path, sub_triplets) 
    
    with open("Process_logs.txt", "a", encoding="utf-8") as file:
        current_time = datetime.now().strftime("%H:%M:%S - %b %d, %Y")
        file.write(f"-----Log at {current_time} ----- ({args.save_path})")
        file.write(f"Using file: {args.path}, saving to: {args.save_path}...\n")

        No_relations = len(list(set([e[1] for e in original_triplets])))
        No_entity = len(list(set([e[0] for e in original_triplets] + [e[2] for e in original_triplets])))
        file.write(f"ALL Numnber of entity: {No_entity}, ALL number of relations: {No_relations}, ALL number of triplets: {len(original_triplets)}\n")
        
        No_relations = len(list(set([e[1] for e in sub_triplets])))
        No_entity = len(list(set([e[0] for e in sub_triplets] + [e[2] for e in sub_triplets])))
        file.write(f"SAMPLE Numnber of entity: {No_entity}, SAMPLE number of relations: {No_relations}, SAMPLE number of triplets: {len(sub_triplets)}\n")
        
        file.write(f"\nStatistic information of each sampled subgraph:")
        for i, e in enumerate(statistics):
            file.write(f"Sampled subgraph No.{i}")
            for key, value in e.items():
                file.write(f"\t{key}:{value}\n")
                
        file.write("\n\n")

    


