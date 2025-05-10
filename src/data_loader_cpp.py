from torch.utils.data import Dataset
from pymongo import MongoClient
import torch
import datetime
import random
from torch.utils.data import Dataset
import torch
 
class Horusec_CPP_Dataset(Dataset):

    def __init__(self, label_type="vuln_score"):
        self.client = MongoClient("mongodb://localhost:27017/")
        self.horusec_cpp_db = self.client["horusec_cpp_val"]
        self.collection_names = self.horusec_cpp_db.list_collection_names()
        self.vulnerability_counts = {}
        self.label_type = label_type
        for conf in ['HIGH', 'MEDIUM', 'LOW']:
            for sev in ['CRITICAL','HIGH', 'MEDIUM', 'LOW']:
                self.vulnerability_counts[(conf, sev)] = 0

    def __len__(self):
        return len(self.horusec_cpp_db.list_collection_names())

    def __getitem__(self, name) -> list:
        return [sorted(self.horusec_cpp_db.list_collection_names())[name], self.get_vulnerability_stats(sorted(self.horusec_cpp_db.list_collection_names())[name])]
    
    def get_tensor(self, name):
        if self.label_type == "vuln_score":
            return self.get_vuln_score(name)
        elif self.label_type == "vuln_count":
            return self.get_vuln_count(name)
        elif self.label_type == "is_vulnerable":
            return self.get_is_vulnerable(name)
        
    def get_vuln_score(self, name):
        return torch.tensor(self._get_vulnerability_score(name), dtype=torch.float32)

    def get_vuln_count(self, name):
        result = list(self.horusec_cpp_db.get_collection(name).find({}))
        if not result:
            return torch.tensor(0, dtype=torch.float32)
        
        total_vulns = 0
        for doc in result:
            if 'results' in doc and 'analysisVulnerabilities' in doc['results']:
                if doc['results']['analysisVulnerabilities'] is not None:
                    total_vulns += len(doc['results']['analysisVulnerabilities'])
        
        return torch.tensor(total_vulns, dtype=torch.float32)

    def get_vuln_score_detailed(self, name):
        return self.get_vulnerability_stats(name)
    
    def get_is_vulnerable(self, name):
        if self.get_vuln_count(name) > 10:
            return torch.tensor(1, dtype=torch.float32)
        else:
            return torch.tensor(0, dtype=torch.float32)

    def get_vulnerability_stats(self, collection_name) -> list[dict[str, int]]:
        for key in self.vulnerability_counts:
            self.vulnerability_counts[key] = 0
            
        result = list(self.horusec_cpp_db.get_collection(collection_name).find({}))
        if not result:
            return self.convert_vulnerability_dict_to_tensor(self.vulnerability_counts)
        
        for doc in result:
            if 'results' in doc and 'analysisVulnerabilities' in doc['results']:
                if doc['results']['analysisVulnerabilities'] is not None:
                    for vuln_entry in doc['results']['analysisVulnerabilities']:
                        if vuln_entry is not None and 'vulnerabilities' in vuln_entry:
                            vuln = vuln_entry['vulnerabilities']
                            if vuln is not None and 'confidence' in vuln and 'severity' in vuln:
                                key = (vuln['confidence'], vuln['severity'])
                                if key in self.vulnerability_counts:
                                    self.vulnerability_counts[key] += 1

        return self.convert_vulnerability_dict_to_tensor(self.vulnerability_counts)
    
    def _get_vulnerability_score(self, collection_name):

        for key in self.vulnerability_counts:
            self.vulnerability_counts[key] = 0
            
        result = list(self.horusec_cpp_db.get_collection(collection_name).find({}))
        if not result:
            return 0
        
        for doc in result:
            if 'results' in doc and 'analysisVulnerabilities' in doc['results']:
                if doc['results']['analysisVulnerabilities'] is not None:
                    for vuln_entry in doc['results']['analysisVulnerabilities']:
                        if vuln_entry is not None and 'vulnerabilities' in vuln_entry:
                            vuln = vuln_entry['vulnerabilities']
                            if vuln is not None and 'confidence' in vuln and 'severity' in vuln:
                                key = (vuln['confidence'], vuln['severity'])
                                if key in self.vulnerability_counts:
                                    self.vulnerability_counts[key] += 1
        
        severity_values = {'CRITICAL': 3, 'HIGH': 2, 'MEDIUM': 1, 'LOW': 0}
        total_score = 0
        total_count = 0
        
        for (conf, sev), count in self.vulnerability_counts.items():
            if count > 0:
                entry_score = (severity_values[conf] + severity_values[sev]) / 2
                total_score += entry_score * count
                total_count += count
        
        if total_count == 0:
            return 0

        return total_score / total_count
    
    def convert_vulnerability_dict_to_tensor(self, vulnerability_counts):

        severity_to_idx = {"CRITICAL": 0, 'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}

        vulnerability_tensor = torch.zeros(3, 3)

        for (conf, sev), count in vulnerability_counts.items():
            conf_idx = severity_to_idx[conf]
            sev_idx = severity_to_idx[sev]

            vulnerability_tensor[conf_idx, sev_idx] = count

        return vulnerability_tensor
 
       
class Semgrep_CPP_Dataset(Dataset):

    def __init__(self, label_type="vuln_score"):
        self.client = MongoClient("mongodb://localhost:27017/")
        self.semgrep_cpp_db = self.client["semgrep_cpp_tmp_val"]
        self.vulnerability_counts = self._init_dict()
        self.label_type = label_type

    def __len__(self):
        return len(self.semgrep_cpp_db.list_collection_names())

    def __getitem__(self, index) -> list:
        return [sorted(self.semgrep_cpp_db.list_collection_names())[index], self.get_vulnerability_stats(sorted(self.semgrep_cpp_db.list_collection_names())[index])]

    def _init_dict(self):
        vulnerability_counts = {}
        for conf in ['HIGH', 'MEDIUM', 'LOW']:
            for imp in ['HIGH', 'MEDIUM', 'LOW']:
                for lik in ['HIGH', 'MEDIUM', 'LOW']:
                    vulnerability_counts[(conf, imp, lik)] = 0
        return vulnerability_counts
    
    def get_tensor(self, name):
        if self.label_type == "vuln_score":
            return self.get_vuln_score(name)
        elif self.label_type == "vuln_count":
            return self.get_vuln_count(name)
        elif self.label_type == "vuln_score_detailed":
            return self.get_vuln_score_detailed(name)
        elif self.label_type == "is_vulnerable":
            return self.get_is_vulnerable(name)
        
    def get_vuln_score(self, name):
        return torch.tensor(self._get_vulnerability_score(name), dtype=torch.float32)

    def get_vuln_count(self, name):
        return torch.tensor(len(list(self.semgrep_cpp_db.get_collection(name).find({}))), dtype=torch.float32)

    def get_vuln_score_detailed(self, name):
        return self.get_vulnerability_stats(name)
    
    def get_is_vulnerable(self, name):
        if self.get_vuln_count(name) > 10:
            return torch.tensor(1, dtype=torch.float32)
        else:
            return torch.tensor(0, dtype=torch.float32)
    
    def convert_vulnerability_dict_to_tensor(self, vulnerability_counts):

        severity_to_idx = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}

        vulnerability_tensor = torch.zeros(3, 3, 3)

        for (conf, imp, lik), count in vulnerability_counts.items():
            conf_idx = severity_to_idx[conf]
            imp_idx = severity_to_idx[imp]
            lik_idx = severity_to_idx[lik]

            vulnerability_tensor[conf_idx, imp_idx, lik_idx] = count

        return vulnerability_tensor
    
    def calculate_vulnerability_score(self):

        severity_values = {'HIGH': 2, 'MEDIUM': 1, 'LOW': 0}
        total_score = 0
        total_count = 0
        
        for (conf, imp, lik), count in self.vulnerability_counts.items():
            if count > 0:
                entry_score = (severity_values[conf] + severity_values[imp] + severity_values[lik]) / 3
                total_score += entry_score * count
                total_count += count
        
        if total_count == 0:
            return 0
            
        return total_score / total_count

    def get_vulnerability_stats(self, collection_name) -> list[dict[str, int]]:
        for key in self.vulnerability_counts:
            self.vulnerability_counts[key] = 0
        try:
            result = list(self.semgrep_cpp_db.get_collection(collection_name).find({}))
            if not result:
                return self.convert_vulnerability_dict_to_tensor(self.vulnerability_counts)
                
            for vuln in result:
                key = (vuln.get('confidence', 'MEDIUM'), vuln.get('impact', 'MEDIUM'), vuln.get('likelihood', 'MEDIUM'))
                if key in self.vulnerability_counts:
                    self.vulnerability_counts[key] += 1
        except Exception as e:
            pass
            
        return self.convert_vulnerability_dict_to_tensor(self.vulnerability_counts)
    
    def _get_vulnerability_score(self, collection_name):
        for key in self.vulnerability_counts:
            self.vulnerability_counts[key] = 0
            
        try:
            result = list(self.semgrep_cpp_db.get_collection(collection_name).find({}))
            if not result:
                return 0
                
            for vuln in result:
                key = (vuln.get('confidence', 'MEDIUM'), vuln.get('impact', 'MEDIUM'), vuln.get('likelihood', 'MEDIUM'))
                if key in self.vulnerability_counts:
                    self.vulnerability_counts[key] += 1
        except Exception as e:
            return 0
            
        return self.calculate_vulnerability_score()

class Repo_CPP_Dataset(Dataset):
    def __init__(self, sast_select="semgrep", split=0.8, type="train", label_select="vuln_score"):
        self.client = MongoClient("mongodb://localhost:27017/")
        self.repos_cpp_tensors = self.client["repo_tensors"].get_collection("cpp_tensors")
        self.repos_cpp_db = self.client["repos_cpp_val"]
        self.data = self._load_data(split, type)
        self.collection_names = list(self.data.keys())
        self.feature_tensors = {}
        self.features = ['name', 'branches', 'commits', 'issues', 'pulls', 'contributors', 'languages', 'closed_issues', 'forks', 'repo_info', 'subscribers', 'releases', 'issue_comments', 'pull_comments', 'collaborators']
        self.sast_select = sast_select
        self.label_select = label_select

    def _load_data(self, split, type):
        data = {}
        collection_names = [doc['name'] for doc in self.repos_cpp_db.get_collection("repositories").find({}, {"name": 1})]
        if type == "train":
            for collection_name in collection_names[:int(len(collection_names)*split)]:
                data[collection_name] = self.repos_cpp_db.get_collection("repositories").find({"name": collection_name})
        else:
            for collection_name in collection_names[int(len(collection_names)*split):]:
                data[collection_name] = self.repos_cpp_db.get_collection("repositories").find({"name": collection_name})

        return data

    def __len__(self):
        return len(self.data)

    def _get_available_features(self):
        return self.features

    def _pre_calculate_features(self):
        for name in self.data.keys():
            self.feature_tensors[name] = self._create_tensor(name, self._get_available_features())

    def _get_feature_value(self, name, feature):
        try:
            return len(self.data[name][0][feature])
        except (KeyError, IndexError):
            return -1

    def _create_tensor(self, name, features):
        values = [self._get_feature_value(name, feature) for feature in features]
        return torch.tensor(values, dtype=torch.float32)

    def __getitem__(self, index, features=None):

        name = list(self.data.keys())[index]
        if features is None:
            features = self._get_available_features()
        else:
            available = set(self._get_available_features())
            features = [f for f in features if f in available]
            if not features:
                raise ValueError("No valid features specified")

        if self.sast_select == "semgrep":
            label = Semgrep_CPP_Dataset(self.label_select)
        else:
            label = Horusec_CPP_Dataset(self.label_select)

        try:
            tensor = torch.tensor(self.repos_cpp_tensors.find_one({"repo_name": name})["tensor_values"], dtype=torch.float32)
        except (TypeError, KeyError):
            tensor = self._create_tensor(name, features)
            
        label = label.get_tensor(name.split("/")[1])
        return tensor, label

class Commit_History_CPP_Dataset(Dataset):

    def __init__(self, label_select, split, type):
        self.client = MongoClient("mongodb://localhost:27017/", maxPoolSize=None)
        self.commit_history_db = self.client["cpp_commits_val"]
        self.seed = 42
        self.collection_names = self.commit_history_db.list_collection_names()
        self.data = self._load_data(split, type)
        self.label_select = label_select
        self.label = self._select_label()
        self.type = type

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> list:
        name = list(self.data.keys())[index]
        first_commit = self.data[name][0]["author"]["date"].strftime("%Y-%m-%d")
        last_commit = self.data[name][-1]["author"]["date"].strftime("%Y-%m-%d")
        normalized_dates = self.normalize_commit_dates(self.data[name])
        commit_count = len(self.data[name])
        
        if self.label_select == "semgrep":
            label = self.label.get_vuln_score(name)
        elif self.label_select == "is_abandoned":
            label = self.label[name]
        
        return self._create_tensor(first_commit, last_commit, commit_count, normalized_dates), label
    
    def _create_tensor(self, first_commit, last_commit, commit_count, normalized_dates):
        first_date = datetime.datetime.strptime(first_commit, "%Y-%m-%d")
        last_date = datetime.datetime.strptime(last_commit, "%Y-%m-%d")
        
        duration = (last_date - first_date).days
        
        commit_distribution = [normalized_dates.get(i, 0) for i in range(20)]
        
        features = [duration, commit_count] + commit_distribution
        return torch.tensor(features, dtype=torch.float32)

    def _load_data(self, split, type):
        data = {}
        collection_names = self.collection_names.copy()
        random.seed(self.seed)
        random.shuffle(collection_names)
        self.collection_names_train = []
        self.collection_names_test = []
        self.collection_names_validation = []

        if type == "train":
            for collection_name in collection_names[:int(len(collection_names)*split)]:
                data[collection_name] = self.commit_history_db.get_collection(collection_name).find({}).to_list()
                self.collection_names_train.append(collection_name)
        elif type == "test":
            for collection_name in collection_names[int(len(collection_names)*split):]:
                data[collection_name] = self.commit_history_db.get_collection(collection_name).find({}).to_list()
                self.collection_names_test.append(collection_name)

        return data
    
    def _select_label(self):
        if self.label_select == "semgrep":
            return Semgrep_CPP_Dataset("vuln_score")
        elif self.label_select == "is_abandoned":
            return Is_Abandoned_CPP_Dataset()
        elif self.label_select == "horusec":
            return Horusec_CPP_Dataset("vuln_score")
        else:
            return None

    def normalize_commit_dates(self, commit_history):
        if not commit_history:
            return {}
            
        first_date = commit_history[0]["author"]["date"].timestamp()
        last_date = commit_history[-1]["author"]["date"].timestamp()
        total_duration = last_date - first_date
        
        if total_duration == 0:
            return {i: 0 for i in range(20)}
            
        part_duration = total_duration / 20
        commit_counts = {i: 0 for i in range(20)}
        
        for commit in commit_history:
            commit_timestamp = commit["author"]["date"].timestamp()
            part = int((commit_timestamp - first_date) / part_duration)
            if part < 0:
                part = 0
            elif part >= 20:
                part = 19
            commit_counts[part] += 1
        return commit_counts


class Is_Abandoned_CPP_Dataset(Dataset):
    def __init__(self):
        self.client = MongoClient("mongodb://localhost:27017/")
        self.commit_history_db = self.client["cpp_commits_val"]
        self.random_seed = 42
        self.collection_names = self.commit_history_db.list_collection_names()
        self.data = self._load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, name) -> torch.Tensor:
        last_commit_date = self.data[name][0]["author"]["date"]
        current_date = datetime.datetime.now()
        time_diff = current_date - last_commit_date
        days_diff = time_diff.days

        if days_diff > 365:
            return torch.tensor(1.0)
        else:
            return torch.tensor(0.0)
    
    def _load_data(self):
        data = {}
        for collection_name in self.commit_history_db.list_collection_names():
            data[collection_name] = self.commit_history_db.get_collection(collection_name).find({}).to_list()
        return data