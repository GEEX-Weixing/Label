from src.data import Dataset
from src.data_hete import Dataset_2
from src.data_arxiv import get_data_arxiv

def get_datas(dataset_name):
    if dataset_name == 'cora':
        data = Dataset_2('data', dataset=dataset_name)
    elif dataset_name == 'citeseer':
        data = Dataset_2('data', dataset=dataset_name)
    elif dataset_name == 'pubmed':
        data = Dataset_2('data', dataset=dataset_name)
    elif dataset_name == 'computers':
        data = Dataset_2('data', dataset=dataset_name)
    elif dataset_name == 'Coauthor cs':
        data = Dataset_2('data', dataset=dataset_name)
    elif dataset_name == 'arxiv':
        data = get_data_arxiv(dataset_name)
    elif dataset_name == 'texas':
        data = Dataset('data', dataset=dataset_name)
    elif dataset_name == 'wisconsin':
        data = Dataset('data', dataset=dataset_name)
    elif dataset_name == 'chameleon':
        data = Dataset('data', dataset=dataset_name)
    elif dataset_name == 'squirrel':
        data = Dataset('data', dataset=dataset_name)
    elif dataset_name == 'amazon ratings':
        data = Dataset('data', dataset=dataset_name)

    return data







