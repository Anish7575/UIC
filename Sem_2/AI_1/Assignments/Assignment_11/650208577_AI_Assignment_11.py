from locale import currency
from math import log2
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# Tree structure to store the nodes with node test and branches
class Tree:
    root = None
    def __init__(self, node_test):
        self.node_test = node_test
        if self.root == None:
            self.root = self        
        self.attribute_branch = {}
    
    # Helper method to print the decision tree in formatted way
    def printDecisionTree(self):
        cur_node = self.root
        queue = []
        queue.append(cur_node)
        format_string = ''
        while len(queue) > 0:
            node = queue.pop(0)
            print(format_string, '[' + node.node_test + ']')
            format_string += '|---'
            for key, val in node.attribute_branch.items():
                if (type(val) == str):
                    print(format_string, '(' + key + ')', '=>', val)
            
            for key, val in node.attribute_branch.items():
                if (type(val) != str):
                    print(format_string, '(' + key + ')')
                    if val != None:
                        queue.append(val)
            format_string += '|---'
                

# Check the value of delta based on degrees of freedom and remove the
# attribute if the delta is less than the threshold
def check_delta(delta, deg_free):
    if deg_free == 1 and delta < 3.84:
        return True
    if deg_free == 2 and delta < 5.99:
        return True
    if deg_free == 3 and delta < 7.82:
        return True
    return False


# Function implementing chi-square pruning on the decision tree
def chi_square_pruning(root, examples, deg_freedom):
    cur_node = root
    p = examples[label].value_counts().get('Yes', 0)
    n = examples[label].value_counts().get('No', 0)
    delta = 0
    for key, val in cur_node.attribute_branch.items():
        A = cur_node.node_test.split('?')[0]
        attr_val = key.split('=')[1].strip()
        sub_examples = examples[examples[A] == attr_val]
        if len(sub_examples) == 0:
            continue
        # Actual count of examples reaching the node
        p_k = sub_examples[label].value_counts().get('Yes', 0)
        n_k = sub_examples[label].value_counts().get('No', 0)
        # Expected count of examples reaching the node
        pe_k = p * (len(sub_examples) / len(examples))
        ne_k = n * (len(sub_examples) / len(examples))
        if type(val) == str:
            delta += pow(p_k - pe_k, 2)/pe_k + pow(n_k - ne_k, 2)/ne_k
        else:
            tmp_delta = chi_square_pruning(val, sub_examples, deg_freedom)
            attr = val.node_test.split('?')[0]
            # If delta less than threshold attribute is pruned
            if check_delta(tmp_delta, deg_freedom[attr]):
                cur_node.attribute_branch[key] = pluralityValue(sub_examples)

            if type(cur_node.attribute_branch[key]) == str:
                delta += pow(p_k - pe_k, 2)/pe_k + pow(n_k - ne_k, 2)/ne_k
    
    # delta check for root node
    print('Attribute:', cur_node.node_test.split('?')[0], ', Degrees of freedom:', deg_freedom[cur_node.node_test.split('?')[0]], ', Delta:', delta)
    if check_delta(delta, deg_freedom[cur_node.node_test.split('?')[0]]):
        cur_node = None
    
    return delta


# Function to return the dominant label
def pluralityValue(examples):
    return examples[label].mode().values[0]


# Function to check if all the examples have the same label
def checkExampleValues(examples):
    if len(examples) == examples[label].value_counts().get('Yes', 0):
        return 'Yes'
    if len(examples) == examples[label].value_counts().get('No', 0):
        return 'No'
    return []


# Function to calculate the entropy
def calEntropy(p, l):
    q = p/l
    if q == 1.0:
        return -(q*log2(q))
    if q == 0.0:
        return -((1 - q)*log2(1 - q))
    return -(q*log2(q) + (1 - q)*log2(1 - q))


# Function to calculate the remainder
def calRemainder(A, examples):
    vals = examples.groupby(A)[label].value_counts().to_dict()
    rem = 0
    for val in examples[A].unique():
        p_k = 0
        n_k = 0
        if (val, 'Yes') in vals:
            p_k = vals[(val, 'Yes')]
        if (val, 'No') in vals:
            n_k = vals[(val, 'No')]
        rem += (p_k + n_k)/(len(examples)) * calEntropy(p_k, p_k + n_k)
    return rem


# Function to calculate the imprtance of an attribute using
# entropy and remainder
def importance(attribute, examples):
    if 'Yes' in examples[label].value_counts():
        entropy = calEntropy(examples[label].value_counts()['Yes'], len(examples))
    else:
        entropy = calEntropy(0, len(examples))
    remainder = calRemainder(attribute, examples)
    
    return entropy - remainder


# Function to train a decision tree
def learnDecisionTree(examples, attributes, parent_examples, unique_col_vals):
    if len(examples) == 0:
        return pluralityValue(parent_examples)
    cls = checkExampleValues(examples)
    if len(cls) > 0:
        return cls
    elif attributes is None:
        return pluralityValue(examples)
    else:
        A = ''
        info_gain_dict = {}
        for a in attributes[:-1]:
            info_gain_dict[a] = "{:.2f}".format(importance(a, examples))
            
        print(info_gain_dict)
        A = max(info_gain_dict, key=info_gain_dict.get)
        print('Chosen Attribute for the split -> ', A + ': ' + max(info_gain_dict.values()),'\n')
        tree = Tree(A + '?')
        for value in unique_col_vals[A]:
            new_examples = pd.DataFrame()
            # Splitting the data and calling the training function with new data recursively
            for idx, example in examples.iterrows():
                if example[A] == value:
                    new_examples = new_examples.append(example, ignore_index=True)

            subtree = learnDecisionTree(new_examples, [attribute for attribute in attributes if attribute != A], examples, unique_col_vals)
            tree.attribute_branch['A = ' + str(value)] = subtree
        
        return tree


def main():
    url = "https://raw.githubusercontent.com/aimacode/aima-data/master/restaurant.csv"
    # Reading the data into pandas dataframe using the above url
    df = pd.read_csv(url, index_col=False, header=None, delimiter=' *, *', engine='python')
    # Assigning columns to the dataframe
    df.columns = ['Alt', 'Bar', 'Fri', 'Hun', 'Pat', 'Price', 'Rain', 'Res', 'Type', 'Est', 'WillWait']
    global label
    label = df.columns[-1]
    deg_freedom = {}
    unique_col_vals = {}
    for col in df.columns:
        deg_freedom[col] = len(df[col].unique()) - 1
        unique_col_vals[col] = df[col].unique()
    # Training decision tree classifier
    print('Decision Tree classification: \n')
    decision_tree = learnDecisionTree(df, list(df.columns), df, unique_col_vals)
    
    print('\n---------------------------------------------------------------------------------')
    print('Learned Decision Tree for the given data:')
    decision_tree.printDecisionTree()
    print('\n---------------------------------------------------------------------------------')
    # Applying Chi-Square pruning on the decision tree classifier
    print('Chi-Square Pruning: \n')
    chi_square_pruning(decision_tree, df, deg_freedom)
    print('\n')
    print('Decision Tree after applying Chi Square Pruning using 5 percent confidence:')
    decision_tree.printDecisionTree()
        

if __name__ == '__main__':
    main()