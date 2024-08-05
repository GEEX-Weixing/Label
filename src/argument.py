import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", "-r", type=str, default="data")
    parser.add_argument("--dataset", type=str, default="cora", help="cora, citeseer, pubmed, computers, photo")
    
    # masking
    parser.add_argument("--label_rate", type=float, default=0.5)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--edge_range", type=int, default=10)

    # Encoder
    parser.add_argument("--layers", nargs='+', default='[128, 128]', help="The number of units of each layer of the GNN. Default is [256]")
    
    # optimization
    parser.add_argument("--epochs", '-e', type=int, default=400, help="The number of epochs")
    parser.add_argument("--st_epochs", type=int, default=100, help="The epochs of self training")
    parser.add_argument("--lr", '-lr', type=float, default=0.0015, help="Learning rate. Default is 0.0001.")
    parser.add_argument("--decay", type=float, default=1e-5, help="Learning rate. Default is 0.0001.")
    parser.add_argument("--patience", type=int, default=200)

    parser.add_argument("--stage", type=int, default=3)
    parser.add_argument("--device", '-d', type=int, default=5, help="GPU to use")
    parser.add_argument("--pmax", type=float, default=0.95, help="confidence threshold.")

    return parser.parse_known_args()[0]
