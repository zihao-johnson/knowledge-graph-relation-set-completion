from operator import itemgetter
import random
from datetime import datetime
import math
import argparse

def load_triplets(path):
    with open(path, "r") as file:
        triplets = [e.strip().split("\t") for e in file.readlines()]
    return triplets

def connected_relation_analysis(triplets):
    dic = {}
    dic_re = {}
    for line in triplets:
        for index in [0,2]:
            try:
                dic[line[index]]
            except:
                dic[line[index]] = {}
            try:
                dic[line[index]][line[1]] +=1
            except:
                dic[line[index]][line[1]] = 1
        try:
            dic_re[line[1]] += 1
        except:
            dic_re[line[1]] = 1
    dic = {key: dict(sorted(dic[key].items(), key=itemgetter(1), reverse = True)) for key in dic.keys()}
    return dic, dic_re

def generate_train_val_test_data(dic):
    train_set = {}
    val_set = {}
    rest_test = {}

    train_triplets = {}
    valid_triplets = {}
    test_triplets = {}

    ratio = 0.2
    for entity, relations in dic.items():
        if len(relations.keys()) > 2 and len(relations.keys()) <5:
            train_set[entity] = list(relations.keys())
        elif len(relations.keys()) <= 2 :
            rest_test[entity] = list(relations.keys())
        else:
            relation_No = len(list(relations.keys()))
            samples = random.sample(list(relations.keys()), math.ceil(ratio*relation_No))
            try:
                assert samples, "samples is null"
            except:
                print("relation_No:", relation_No)
            val_set[entity] = samples
            train_set[entity] = list(set(relations.keys()).difference(set(val_set[entity])))

    test_set = dict(random.sample(list(val_set.items()),len(list(val_set.items()))//2))
    val_set = dict([(e, val_set[e]) for e in set(val_set.keys()).difference(set(test_set.keys()))])
    return train_set, val_set, test_set, rest_test

def save_file(file_name, data):
    data = ["\t".join([entity] + relations) for entity, relations in data.items()]
    print(len(data))
    with open(file_name, 'w') as the_file:
        the_file.write("\n".join(data))

def statistic(dic):
    dicc = {}
    for e in dic.keys():
        try:
            dicc[len(dic[e].keys())] += 1
        except:
            dicc[len(dic[e].keys())] = 1
    dicc= {key:dicc[key] for key in sorted(dicc.keys())}
    
    return dicc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="My script with defaults")

    parser.add_argument("--path", default="all.txt", help="Input file name")
    parser.add_argument("--save_path", default="log.txt", help="Clearned dataset")
    
    args = parser.parse_args()

    triplets = load_triplets(args.path)
    dic, dic_re = connected_relation_analysis(triplets)
    dic_re = dict(sorted(dic_re.items(), key=itemgetter(1), reverse = True))
    dicc = statistic(dic)
    train_set, val_set, test_set, rest_test = generate_train_val_test_data(dic)
    
    save_file("train.txt", train_set)
    save_file("valid.txt", val_set)
    save_file("test.txt", test_set)
    save_file("rest.txt", rest_test)
    
    with open(args.save_path, "w", encoding="utf-8") as file:
        current_time_24hr = datetime.now().strftime("%H:%M:%S - %b %d, %Y")
        file.write(f"-----Log at {current_time_24hr} ----- ({args.save_path})\n")

        file.write(f"No. of triplets: {len(triplets)}, No. of entities {len(set([triplet[0] for triplet in triplets] + [triplet[2] for triplet in triplets]))}, No. of relations {len(set([triplet[1] for triplet in triplets]))}, No. of train: {len(train_set)}, No. of val: {len(val_set)}, No. of test: {len(test_set)} \n")
        
        file.write(f"relation numbers : entity numbers ({sum(list(dicc.values()))}) \n")
        for key, value in dicc.items():
            file.write(f"{key} : {value}\n")

        file.write(f"relation : appear numbers ({sum(list(dic_re.values()))}) \n")
        for key, value in dic_re.items():
            file.write(f"{key} : {value}\n")
        file.write("\n\n")

    